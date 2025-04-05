import os
import uuid
import shutil
import boto3
import json
import asyncio
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import FastAPI, HTTPException, Query, Response, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncGenerator, Dict, Any
from dotenv import load_dotenv
from datetime import datetime, timezone

# --- Load environment variables from .env file ---
load_dotenv() # <-- Load variables from .env file into the environment

# --- Import from generate.py ---
# (This section should come *after* load_dotenv if generate.py also needs env vars,
# but in this case, only api.py needs the R2 vars directly)
try:
    from generate import text_to_speech, process_conversation_async, AVAILABLE_VOICES
except ImportError:
    print("Error: Could not import functions from generate.py. Make sure it's in the same directory or Python path.")
    # Assign defaults or raise error to prevent app start?
    AVAILABLE_VOICES = []
    async def text_to_speech(*args, **kwargs): raise NotImplementedError("TTS function not loaded")
    async def process_conversation_async(*args, **kwargs): raise NotImplementedError("Async Conversation function not loaded")


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
        await asyncio.to_thread(
            r2_client.upload_file,
            local_file_path,
            R2_BUCKET_NAME,
            object_key,
            ExtraArgs={'ContentType': 'audio/wav', 'ACL': 'public-read'}
        )
        print("Upload successful.")

        # Construct the public URL
        if R2_PUBLIC_URL_BASE:
             public_url = f"{R2_PUBLIC_URL_BASE.rstrip('/')}/{object_key.lstrip('/')}"
        else:
             print("Warning: R2_PUBLIC_URL_BASE not set. Attempting default URL construction.")
             try:
                  endpoint_parts = R2_ENDPOINT_URL.split('.')
                  if len(endpoint_parts) > 2 and '//' in endpoint_parts[0]:
                       account_id = endpoint_parts[0].split('//')[1]
                       public_url_base = f"https://pub-{R2_ENDPOINT_URL.split('.')[0].split('-')[1]}.r2.dev"
                       public_url = f"{public_url_base}/{object_key}"
                       print(f"Constructed default URL: {public_url}. Ensure public access enabled.")
                  else:
                       print("Warning: Could not parse account ID from endpoint URL.")
                       public_url = None
             except Exception as e:
                  print(f"Warning: Failed to construct default public URL: {e}")
                  public_url = None

        return public_url

    except FileNotFoundError:
        print(f"Error: Local file not found for upload: {local_file_path}")
        return None
    except NoCredentialsError:
        print("Error: R2 credentials not found by boto3.")
        return None
    except ClientError as e:
        if e.response['Error']['Code'] == 'AccessDenied':
             print(f"Error uploading to R2: Access Denied.")
        else:
             print(f"Error uploading to R2: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during R2 upload: {e}")
        return None
# --- END R2 UPLOAD FUNCTION ---

# --- NEW: Synchronous SSE Formatter ---
def format_sse(data: dict) -> str:
    """Formats a dictionary into an SSE message string."""
    if not isinstance(data, dict):
        print(f"Warning: Invalid data type passed to format_sse: {type(data)}")
        data = {"status": "internal_warning", "message": "Invalid event data type"}
    # Ensure status exists if only message provided (optional enhancement)
    # if "status" not in data and "message" in data:
    #     data["status"] = "progress"
    return f"data: {json.dumps(data)}\n\n"
# --- END Formatter ---

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
    "/generate-conversation-stream",
    tags=["TTS"],
    summary="Generate conversation audio, upload to R2, and stream progress",
    # Response model is not directly applicable here as it's a stream
    responses={
        200: {"description": "SSE stream started. Events contain progress and final URL/error."},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        501: {"model": ErrorResponse, "description": "Functionality not loaded or R2 not configured initially"},
     }
)
async def generate_conversation_stream(request_data: ConversationRequest): # Use the Pydantic model for validation
    """
    Starts a Server-Sent Events (SSE) stream to report progress on
    generating conversation audio and uploading it to Cloudflare R2.

    Events are JSON strings containing `status` and optional `message`, `progress`,
    `audio_url`, `r2_object_key`, `local_path`.

    Example Event: `data: {"status": "progress", "message": "Generated segment 1/10"}\n\n`
    Final Event: `data: {"status": "success", "audio_url": "...", ...}\n\n` or `data: {"status": "error", ...}\n\n`
    """

    # Define the async generator for SSE events
    async def event_generator() -> AsyncGenerator[str, None]:
        generated_local_path: Optional[str] = None
        request_temp_dir: Optional[str] = None
        r2_full_object_key: Optional[str] = None
        final_result_data: Dict[str, Any] = {} # To store final data for the last event

        # Use the synchronous formatter now
        yield format_sse({"status": "start", "message": "Starting process..."})
        await asyncio.sleep(0.01)

        try:
            # --- 1. Validation ---
            yield format_sse({"status": "validation", "message": "Validating voices..."})
            if not callable(process_conversation_async):
                 raise ValueError("Async conversation processing function not available.")
            for i, line in enumerate(request_data.conversation):
                if line.voice not in AVAILABLE_VOICES:
                    raise HTTPException(status_code=400, detail=f"Invalid voice '{line.voice}' for line {i+1}.")
            yield format_sse({"status": "validation", "message": "Validation complete."})
            await asyncio.sleep(0.01)

            # --- 2. Prepare ---
            local_filename_base = request_data.output_filename_base
            local_audio_filename = generate_safe_filename(local_filename_base)
            local_combined_filepath = os.path.join(API_OUTPUT_DIR, local_audio_filename)
            request_temp_dir = os.path.join(API_TEMP_DIR, str(uuid.uuid4()))
            conversation_data_for_processing = [{"text": l.text, "voice": l.voice, "pause_after": l.pause_after} for l in request_data.conversation]

            yield format_sse({"status": "generation_start", "message": f"Starting generation..."})
            await asyncio.sleep(0.01)

            # --- 3. Run Async Local Generation & Stream Progress ---
            # This generator yields progress updates AND a final result dict
            process_gen = process_conversation_async(
                conversation_data_for_processing,
                local_combined_filepath,
                request_temp_dir
            )
            async for event_data in process_gen:
                 # Check if this is the final result dict from the generator
                 if event_data.get("type") == "result":
                     generator_status = event_data.get("status", "error") # Get status (finished, error, etc.)
                     generated_local_path = event_data.get("path") # Get the final path (or None)
                     print(f"Generator finished with status: {generator_status}, path: {generated_local_path}")
                     if generator_status == "error" or not generated_local_path:
                          # If the generator itself reported a critical failure or no path
                          final_result_data = {
                               "status": "error",
                               "message": event_data.get("message", "Generation process failed internally.")
                          }
                     # Break loop, handle upload/final result below
                     break
                 else:
                     # Forward progress/warning events directly to the client
                     yield format_sse(event_data) # Use formatter
                     await asyncio.sleep(0.01)

            # --- Check if generation was successful (path exists) ---
            if not generated_local_path:
                 # If we broke the loop but still have no path, use the stored error or a default
                 if not final_result_data: # Check if error wasn't already set
                     final_result_data = {
                          "status": "error",
                          "message": "Local conversation generation failed (no output path)."
                     }
                 # Skip upload and proceed to sending the final (error) event

            # --- 4. Upload to R2 (only if generation succeeded) ---
            elif r2_client: # Check if R2 is configured AND generation was ok
                yield format_sse({"status": "upload_start", "message": "Starting R2 upload..."})
                await asyncio.sleep(0.01)
                now_utc = datetime.now(timezone.utc)
                timestamp_folder = now_utc.strftime("output_%Y-%m-%dT%H-%M-%S-%f")[:-3] + "Z"
                audio_file_basename = os.path.basename(generated_local_path)
                r2_full_object_key = f"audio/{timestamp_folder}/{audio_file_basename}"

                audio_url = await upload_to_r2(generated_local_path, r2_full_object_key)

                if audio_url:
                    yield format_sse({"status": "upload_complete", "message": "R2 upload successful."})
                    asyncio.create_task(cleanup_file(generated_local_path)) # Cleanup local file async
                    final_result_data = {
                        "status": "success",
                        "message": "Conversation generated and uploaded successfully.",
                        "audio_url": audio_url,
                        "r2_object_key": r2_full_object_key
                    }
                else: # Upload failed
                    final_result_data = {
                        "status": "error", # Treat upload failure as overall error for this endpoint's goal
                        "message": "Conversation generated locally, but R2 upload failed.",
                        "local_path": generated_local_path, # Keep local file
                        "r2_object_key": r2_full_object_key
                    }
            elif generated_local_path: # Generation succeeded, but R2 not configured
                final_result_data = {
                    "status": "success_local",
                    "message": "Conversation generated locally. R2 upload skipped (not configured).",
                    "local_path": generated_local_path # Keep local file
                }

            # --- Send Final Result Event ---
            if final_result_data: # Ensure we have something to send
                 yield format_sse(final_result_data)
            else: # Should not happen if logic is correct, but as a fallback
                 yield format_sse({"status": "error", "message": "Processing ended in an unexpected state."})


        except HTTPException as e: # Catch validation errors
             yield format_sse({"status": "error", "message": f"Input Error: {e.detail}"})
        except Exception as e: # Catch other unexpected errors
             print(f"Error during conversation stream: {e}", exc_info=True)
             yield format_sse({"status": "error", "message": f"An internal server error occurred: {str(e)}"})
        # Note: Cleanup of request_temp_dir is handled within process_conversation_async

    # Return the StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
