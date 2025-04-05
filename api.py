import os
import uuid
import shutil
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import FastAPI, HTTPException, Query, Response, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv # <-- Import load_dotenv
from datetime import datetime, timezone # <-- Import datetime

# --- Load environment variables from .env file ---
load_dotenv() # <-- Load variables from .env file into the environment

# --- Import from generate.py ---
# (This section should come *after* load_dotenv if generate.py also needs env vars,
# but in this case, only api.py needs the R2 vars directly)
try:
    from generate import text_to_speech, process_conversation, AVAILABLE_VOICES, combine_wav_files
except ImportError:
    print("Error: Could not import functions from generate.py. Make sure it's in the same directory or Python path.")
    # Assign defaults or raise error to prevent app start?
    AVAILABLE_VOICES = []
    async def text_to_speech(*args, **kwargs): raise NotImplementedError("TTS function not loaded")
    async def process_conversation(*args, **kwargs): raise NotImplementedError("Conversation function not loaded")


app = FastAPI(title="Orpheus TTS API", version="0.1.0")

# --- Configuration (Now reads loaded env vars) ---
API_OUTPUT_DIR = os.environ.get("API_OUTPUT_DIR", "api_output")
API_TEMP_DIR = os.environ.get("API_TEMP_DIR", os.path.join(API_OUTPUT_DIR, "temp"))

os.makedirs(API_OUTPUT_DIR, exist_ok=True)
os.makedirs(API_TEMP_DIR, exist_ok=True) # For intermediate files from process_conversation

# --- R2 Configuration (Reads loaded env vars) ---
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_PUBLIC_URL_BASE = os.environ.get("R2_PUBLIC_URL_BASE")

r2_client = None # Initialize r2_client to None
r2_configured = all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME])
if not r2_configured:
    print("Warning: R2 environment variables not fully configured. Upload to R2 will be disabled.")
else:
    try:
        session = boto3.session.Session()
        r2_client = session.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto',
        )
        print(f"R2 Client configured for bucket: {R2_BUCKET_NAME}")
    except Exception as e:
        print(f"Error configuring R2 client: {e}")
        r2_client = None # Ensure it's None on error

# --- Pydantic Models for Request/Response Data ---

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech", min_length=1)
    voice: str = Field(default="tara", description=f"Voice model to use. Available: {', '.join(AVAILABLE_VOICES) or 'None Found'}")

# Use FileResponse directly, but define model for documentation/error cases
class TTSResponse(BaseModel):
    detail: str # For error messages

class ConversationLine(BaseModel):
    text: str = Field(..., description="Text for this line of conversation", min_length=1)
    voice: str = Field(..., description=f"Voice model for this line. Available: {', '.join(AVAILABLE_VOICES) or 'None Found'}")
    pause_after: float = Field(default=0.5, description="Pause duration in seconds after this line (ignored for the last line)", ge=0.0)

class ConversationRequest(BaseModel):
    conversation: List[ConversationLine] = Field(..., description="List of conversation lines", min_items=1)
    output_filename_base: str = Field(default="conversation", description="Base name for the final combined audio file (will have UUID appended)")

# Use FileResponse, define model for docs/errors
class ConversationResponse(BaseModel):
    detail: str

# Updated response model for conversation endpoint
class ConversationUrlResponse(BaseModel):
    message: str
    audio_url: Optional[str] = None
    local_path: Optional[str] = None # Optionally return local path if R2 fails/disabled
    r2_object_key: Optional[str] = None # Add the key for reference

# Error response model (can reuse ConversationResponse or TTSResponse)
class ErrorResponse(BaseModel):
    detail: str

# --- Helper Functions ---

def generate_safe_filename(base: str, extension: str = ".wav") -> str:
    """Generates a unique, safe filename."""
    safe_base = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base)
    return f"{safe_base}_{uuid.uuid4()}{extension}"

async def cleanup_file(file_path: Optional[str]):
    """Removes a file in the background, checks existence."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Cleaned up file: {file_path}")
        except OSError as e:
            print(f"Error cleaning up file {file_path}: {e}")
    # else:
    #     print(f"Cleanup skipped, file not found or None: {file_path}")

# --- BEGIN R2 UPLOAD FUNCTION ---
async def upload_to_r2(local_file_path: str, object_key: str) -> Optional[str]:
    """Uploads a file to R2 and returns the public URL."""
    if not r2_client:
        print("R2 client not configured. Skipping upload.")
        return None

    try:
        print(f"Uploading {local_file_path} to R2 bucket '{R2_BUCKET_NAME}' as '{object_key}'...")
        r2_client.upload_file(
            local_file_path,
            R2_BUCKET_NAME,
            object_key,
            ExtraArgs={'ContentType': 'audio/wav', 'ACL': 'public-read'} # Ensure public read ACL
        )
        print("Upload successful.")

        # Construct the public URL
        if R2_PUBLIC_URL_BASE:
             # Ensure no double slashes if base URL already has one at the end
             public_url = f"{R2_PUBLIC_URL_BASE.rstrip('/')}/{object_key.lstrip('/')}"
        else:
             # Attempt to construct default URL (less reliable, needs public bucket access)
             print("Warning: R2_PUBLIC_URL_BASE not set. Attempting default URL construction (requires public bucket).")
             try:
                  endpoint_parts = R2_ENDPOINT_URL.split('.')
                  if len(endpoint_parts) > 2 and '//' in endpoint_parts[0]:
                       account_id = endpoint_parts[0].split('//')[1]
                       # Using pub-<hash>.r2.dev format (common for public R2 buckets)
                       # This requires enabling the public bucket URL feature in Cloudflare R2 settings
                       public_url_base = f"https://pub-{R2_ENDPOINT_URL.split('.')[0].split('-')[1]}.r2.dev" # Heuristic, might need adjustment
                       public_url = f"{public_url_base}/{object_key}"
                       print(f"Constructed default URL: {public_url}. Ensure bucket public access is enabled at this base URL.")
                  else:
                       print("Warning: Could not parse account ID from endpoint URL. Cannot construct default public URL.")
                       public_url = None
             except Exception as e:
                  print(f"Warning: Failed to parse endpoint URL or construct default public URL: {e}")
                  public_url = None

        return public_url

    except FileNotFoundError:
        print(f"Error: Local file not found for upload: {local_file_path}")
        return None
    except NoCredentialsError:
        print("Error: R2 credentials not found by boto3.")
        return None
    except ClientError as e:
        # Check for permission errors specifically
        if e.response['Error']['Code'] == 'AccessDenied':
             print(f"Error uploading to R2: Access Denied. Check bucket permissions and credentials.")
        else:
             print(f"Error uploading to R2: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during R2 upload: {e}")
        return None
# --- END R2 UPLOAD FUNCTION ---


# --- API Endpoints ---

@app.get("/voices", tags=["Info"], summary="List available TTS voices")
async def get_available_voices():
    """Returns a list of available voice models loaded by the underlying TTS library."""
    if not AVAILABLE_VOICES:
         return {"message": "No voices loaded or found by the TTS library.", "available_voices": []}
    return {"available_voices": AVAILABLE_VOICES}

@app.post(
    "/generate",
    tags=["TTS"],
    summary="Generate speech for a single text input",
    response_class=FileResponse, # Directly return the file
    responses={ # Define potential error responses
        200: {
            "content": {"audio/wav": {}},
            "description": "Successful TTS Generation. Returns the WAV file.",
        },
        400: {"model": TTSResponse, "description": "Invalid input (e.g., unknown voice)"},
        500: {"model": TTSResponse, "description": "Internal server error during generation"},
        501: {"model": TTSResponse, "description": "TTS functionality not loaded"},
     }
)
async def generate_single_tts(
    request: TTSRequest,
    background_tasks: BackgroundTasks,
    cleanup: bool = Query(True, description="Delete the generated file from the server after it's sent?")
):
    """
    Converts the provided text to speech using the specified voice.

    - **text**: The text content to synthesize.
    - **voice**: The voice model to use. Check the `/voices` endpoint for options.
    - **cleanup**: If true (default), the generated audio file is deleted from the server after being sent. Set to false to keep the file temporarily on the server (within `API_OUTPUT_DIR`).
    """
    # Check if TTS function is loaded (handles import error case)
    if not callable(text_to_speech):
        raise HTTPException(status_code=501, detail="TTS function (text_to_speech) is not available.")

    if request.voice not in AVAILABLE_VOICES:
        raise HTTPException(status_code=400, detail=f"Voice '{request.voice}' not available. Available voices: {AVAILABLE_VOICES}")

    # Generate a unique filename in the main output directory
    output_filename = generate_safe_filename(f"tts_{request.voice}")
    output_path = os.path.join(API_OUTPUT_DIR, output_filename)

    try:
        print(f"API generating single TTS: voice='{request.voice}', output='{output_path}'")
        # Call the imported function from generate.py
        # Assuming it writes the file directly if output_file is provided
        text_to_speech(
            text=request.text,
            voice=request.voice,
            output_file=output_path
        )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
             print(f"TTS generation failed or produced empty file: {output_path}")
             # Attempt cleanup just in case an empty file was made
             await cleanup_file(output_path)
             raise HTTPException(status_code=500, detail="TTS generation failed to create a valid output file.")

        print(f"TTS generation successful: {output_path}")

        # Add cleanup task if requested
        cleanup_task = cleanup_file if cleanup else None # Pass None to FileResponse if no cleanup

        # Return the file as a response
        return FileResponse(
            path=output_path,
            media_type='audio/wav',
            filename=output_filename, # Suggests a filename to the client
            background=cleanup_task # Use FileResponse's background cleanup
        )

    except HTTPException:
        raise # Re-raise HTTP exceptions directly
    except Exception as e:
        # Clean up the potentially partially created file on error
        await cleanup_file(output_path)
        print(f"Error during TTS generation: {e}", exc_info=True) # Log traceback
        raise HTTPException(status_code=500, detail=f"An internal error occurred during TTS generation: {str(e)}")


@app.post(
    "/generate-conversation",
    tags=["TTS"],
    summary="Generate conversation audio, upload to R2, and return URL",
    response_model=ConversationUrlResponse,
    responses={
        200: {"description": "Successful. Returns JSON with the audio URL or local path info."},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        501: {"model": ErrorResponse, "description": "Functionality not loaded or R2 not configured"},
     }
)
async def generate_conversation_r2(
    request: ConversationRequest,
):
    """
    Generates conversation audio, uploads it to Cloudflare R2 (if configured)
    within a nested timestamped folder structure (audio/output_YYYY-MM-DDTHH-MM-SS-msZ/),
    and returns the public URL.

    - **conversation**: List of objects with `text`, `voice`, `pause_after`.
    - **output_filename_base**: Base name for the audio file itself.
    """
    if not callable(process_conversation):
        raise HTTPException(status_code=501, detail="Conversation processing function not available.")

    for i, line in enumerate(request.conversation):
        if line.voice not in AVAILABLE_VOICES:
            raise HTTPException(status_code=400, detail=f"Voice '{line.voice}' for line {i+1} not available.")

    local_filename_base = request.output_filename_base
    local_audio_filename = generate_safe_filename(local_filename_base)
    local_combined_filepath = os.path.join(API_OUTPUT_DIR, local_audio_filename)
    request_temp_dir = os.path.join(API_TEMP_DIR, str(uuid.uuid4()))

    conversation_data_for_processing = [
        {"text": line.text, "voice": line.voice, "pause_after": line.pause_after}
        for line in request.conversation
    ]

    generated_local_path: Optional[str] = None
    r2_full_object_key: Optional[str] = None

    try:
        print(f"API generating conversation locally, target: {local_combined_filepath}, temp dir: {request_temp_dir}")
        generated_local_path = process_conversation(
            conversation_data=conversation_data_for_processing,
            combined_filename=local_combined_filepath,
            intermediate_dir=request_temp_dir
        )

        if not generated_local_path or not os.path.exists(generated_local_path):
             print(f"Local conversation generation failed or file not found: {local_combined_filepath}")
             raise HTTPException(status_code=500, detail="Local conversation generation failed.")

        print(f"Local conversation generation successful: {generated_local_path}")

        if r2_client:
            now_utc = datetime.now(timezone.utc)
            timestamp_folder = now_utc.strftime("output_%Y-%m-%dT%H-%M-%S-%f")[:-3] + "Z"
            audio_file_basename = os.path.basename(generated_local_path)

            r2_full_object_key = f"audio/{timestamp_folder}/{audio_file_basename}"

            print(f"Attempting upload to R2 with key: {r2_full_object_key}") # Log the key being used
            audio_url = await upload_to_r2(generated_local_path, r2_full_object_key)

            if audio_url:
                await cleanup_file(generated_local_path)
                return JSONResponse(content={
                    "message": "Conversation generated and uploaded to R2 successfully.",
                    "audio_url": audio_url,
                    "r2_object_key": r2_full_object_key
                })
            else:
                 return JSONResponse(status_code=500, content={
                    "message": "Conversation generated locally, but R2 upload failed.",
                    "audio_url": None,
                    "local_path": generated_local_path,
                    "r2_object_key": r2_full_object_key
                 })
        else:
            # R2 not configured
            print("R2 not configured. Returning local path info.")
            return JSONResponse(status_code=200, content={ # 200 OK, but indicate skip
                "message": "Conversation generated locally. R2 upload skipped (not configured).",
                "audio_url": None,
                "local_path": generated_local_path
            })

    except HTTPException as e:
        if generated_local_path and os.path.exists(generated_local_path):
             await cleanup_file(generated_local_path)
        if os.path.exists(request_temp_dir):
             try: shutil.rmtree(request_temp_dir)
             except OSError: pass
        raise e
    except Exception as e:
        if generated_local_path and os.path.exists(generated_local_path):
            await cleanup_file(generated_local_path)
        if os.path.exists(request_temp_dir):
             try: shutil.rmtree(request_temp_dir)
             except OSError: pass
        print(f"Error during conversation generation/upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- Run the API (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    # No need to manually set env vars here if using .env
    print(f"--- Starting Orpheus TTS FastAPI Server ---")
    print(f"Attempting to load environment variables from .env file...")
    # load_dotenv() # Load again here if needed, but loading at top is usually sufficient
    print(f"Voices loaded: {', '.join(AVAILABLE_VOICES) or 'None Found'}")
    print(f"API Output Dir : {os.path.abspath(API_OUTPUT_DIR)}")
    print(f"API Temp Dir   : {os.path.abspath(API_TEMP_DIR)}")
    # Re-check R2 config status - simplified check here based on r2_client itself
    # The r2_client variable is defined above, it will be None if config/init failed
    if r2_client: # <-- Simplified check: If r2_client exists and is not None
        print(f"R2 Upload     : Enabled (Bucket: {R2_BUCKET_NAME})")
    else:
        # Check the configuration variables again for a more specific message if disabled
        r2_configured_check = all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME])
        if not r2_configured_check:
            print(f"R2 Upload     : Disabled (Check .env file - missing required variables)")
        else:
             print(f"R2 Upload     : Disabled (R2 client initialization failed - check credentials/endpoint in .env or logs)")
    print(f"Access API docs at http://127.0.0.1:8000/docs")
    print(f"-------------------------------------------")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
