#!/usr/bin/env python3
"""
Reusable functions for using Orpheus TTS and example usage,
including an async version for progress streaming.
"""
import wave
import os
import time
import sys
import shutil
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Any

# Add the directory containing this file (generate.py) to the Python path.
# This helps Python find gguf_orpheus.py and decoder.py if they are
# in the same directory, especially when run via Uvicorn.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    # Optional: Add a print statement for debugging if needed
    # print(f"[generate.py] Added {script_dir} to sys.path")

# Now try importing gguf_orpheus
try:
    from gguf_orpheus import generate_speech_from_api, AVAILABLE_VOICES
except ImportError as e:
     # Add more debug info if the import still fails
     print(f"ERROR in generate.py: Failed to import from gguf_orpheus.")
     print(f"Attempted import from sys.path: {sys.path}")
     print(f"Original error: {e}")
     raise # Re-raise the error to stop execution

# --- Synchronous Core Functions ---

def text_to_speech(text: str, voice: str = "tara", output_file: Optional[str] = None) -> Any:
    """
    Synchronously convert text to speech using Orpheus TTS.

    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (default: tara)
        output_file (str): Path to save the audio file (default: None)

    Returns:
        Any: Result from generate_speech_from_api (likely None if output_file used).
    """
    print(f"[generate.py] Converting (sync): '{text[:50]}...' with voice '{voice}'")
    result = generate_speech_from_api(
        prompt=text,
        voice=voice,
        output_file=output_file
    )
    return result

def combine_wav_files(input_files: List[str], output_file: str, pause_durations: Optional[List[float]] = None) -> bool:
    """
    Synchronously combines multiple WAV files into a single WAV file.
    Returns True on success, False on failure.
    """
    if not input_files:
        print("[generate.py] No input files provided for combining.")
        return False

    num_files = len(input_files)
    if pause_durations is not None and len(pause_durations) != num_files - 1:
        print(f"[generate.py] Error: Pause durations length ({len(pause_durations)}) mismatch with input files ({num_files}).")
        return False

    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    audio_data_list = []
    params = None
    valid_files_count = 0

    # First pass: Read valid files
    for filename in input_files:
        try:
            with wave.open(filename, 'rb') as wf:
                current_params = wf.getparams()
                if params is None:
                    params = current_params
                elif params[:3] != current_params[:3]:
                     print(f"[generate.py] Warning: Skipping {filename}. Incompatible parameters.")
                     continue
                audio_data_list.append(wf.readframes(wf.getnframes()))
                valid_files_count += 1
        except FileNotFoundError:
            print(f"[generate.py] Warning: File not found {filename}. Skipping.")
        except wave.Error as e:
            print(f"[generate.py] Warning: Error reading {filename}: {e}. Skipping.")

    if not audio_data_list or params is None:
        print("[generate.py] No valid audio data found for combining.")
        return False

    # Adjust pauses
    adjusted_pause_durations = None
    if pause_durations is not None:
        if valid_files_count != num_files:
            print(f"[generate.py] Warning: Adjusting pauses due to skipped files.")
            if len(pause_durations) >= valid_files_count - 1:
                 adjusted_pause_durations = pause_durations[:valid_files_count - 1]
            else:
                 adjusted_pause_durations = pause_durations # May be inaccurate
        else:
             adjusted_pause_durations = pause_durations

    # Second pass: Write output file
    try:
        with wave.open(output_file, 'wb') as wf:
            wf.setparams(params)
            nchannels, sampwidth, framerate, _, _, _ = params
            for i, audio_data in enumerate(audio_data_list):
                if i > 0 and adjusted_pause_durations is not None:
                    if i - 1 < len(adjusted_pause_durations):
                        current_pause = adjusted_pause_durations[i-1]
                        if current_pause > 0:
                            wf.writeframes(b'\x00' * int(current_pause * framerate * nchannels * sampwidth))
                wf.writeframes(audio_data)
        print(f"[generate.py] Successfully combined {valid_files_count} files into {output_file}")
        return True
    except Exception as e:
         print(f"[generate.py] Error writing combined WAV file {output_file}: {e}")
         return False


def process_conversation(conversation_data: List[Dict], combined_filename: str, intermediate_dir: str = "temp_audio") -> Optional[str]:
    """
    Synchronously generates, combines, and cleans up conversation audio.

    Returns: Absolute path to the combined file if successful, None otherwise.
    """
    os.makedirs(intermediate_dir, exist_ok=True)
    generated_files = []
    total_lines = len(conversation_data)

    # Generate segments
    print("[generate.py] Starting synchronous generation...")
    for i, line in enumerate(conversation_data, start=1):
        print(f"--- Generating segment {i}/{total_lines} ---")
        segment_filename = f"segment_{i}_{line.get('voice', 'default')}_{int(time.time() * 1000)}.wav"
        segment_filepath = os.path.join(intermediate_dir, segment_filename)
        try:
             text_to_speech(text=line["text"], voice=line["voice"], output_file=segment_filepath)
             if os.path.exists(segment_filepath) and os.path.getsize(segment_filepath) > 0:
                 generated_files.append(segment_filepath)
             else:
                 print(f"-> Failed: File {segment_filename} empty/not created.")
                 if os.path.exists(segment_filepath): os.remove(segment_filepath)
        except Exception as e:
             print(f"-> Error generating {segment_filename}: {e}")
             if os.path.exists(segment_filepath): os.remove(segment_filepath)
    print(f"[generate.py] Synchronous generation complete. Generated {len(generated_files)}/{total_lines} segments.")

    # Combine segments
    final_combined_path = None
    if generated_files:
        print("[generate.py] Starting combination...")
        pause_durations = [line["pause_after"] for line in conversation_data[:-1]] if len(generated_files) == total_lines else None
        if len(generated_files) == 1:
             try:
                 output_dir = os.path.dirname(os.path.abspath(combined_filename))
                 if output_dir: os.makedirs(output_dir, exist_ok=True)
                 shutil.copy2(generated_files[0], combined_filename)
                 final_combined_path = os.path.abspath(combined_filename)
                 print("[generate.py] Copied single segment.")
             except Exception as e: print(f"[generate.py] Error copying single segment: {e}")
        elif len(generated_files) > 1:
            if combine_wav_files(generated_files, combined_filename, pause_durations=pause_durations):
                final_combined_path = os.path.abspath(combined_filename)
            else:
                 print("[generate.py] Combination failed.")
    else:
         print("[generate.py] Skipping combination.")

    # Cleanup
    if generated_files:
        print("[generate.py] Cleaning up intermediate files...")
        deleted_count = 0
        for f in generated_files:
            try:
                if os.path.exists(f): os.remove(f); deleted_count += 1
            except OSError as e: print(f"Error deleting {f}: {e}")
        print(f"[generate.py] Cleanup complete. Deleted {deleted_count} files.")
        try:
             if not os.listdir(intermediate_dir): os.rmdir(intermediate_dir)
        except OSError as e: print(f"Could not remove dir {intermediate_dir}: {e}")

    return final_combined_path

# --- Async Version for Streaming ---

async def process_conversation_async(conversation_data: List[Dict], combined_filename: str, intermediate_dir: str = "temp_audio") -> AsyncGenerator[Dict, None]:
    """
    Asynchronously generates individual audio files for a conversation, yielding progress,
    combines them, and cleans up intermediate files.

    Yields: Progress dictionaries. Final yield is {"type": "result", "status": "...", "path": ...}.
    """
    os.makedirs(intermediate_dir, exist_ok=True)
    generated_files = []
    total_lines = len(conversation_data)
    total_generation_time = 0.0
    final_combined_path = None
    has_errors = False # Track if any segment failed

    yield {"type": "progress", "status": "start", "message": f"Starting generation of {total_lines} segments.", "current": 0, "total": total_lines}

    # --- Generate segments asynchronously ---
    try:
        for i, line in enumerate(conversation_data, start=1):
            status_data = {"current": i-1, "total": total_lines} # Progress before starting
            yield {"type": "progress", "status": "segment_start", "message": f"Generating segment {i}/{total_lines}...", **status_data}

            segment_filename = f"segment_{i}_{line.get('voice', 'default')}_{int(time.time() * 1000)}.wav"
            segment_filepath = os.path.join(intermediate_dir, segment_filename)
            start_time = time.monotonic()
            success = False
            error_message = None

            try:
                # Run blocking text_to_speech in thread
                await asyncio.to_thread(
                    text_to_speech, text=line["text"], voice=line["voice"], output_file=segment_filepath
                )
                end_time = time.monotonic()
                if os.path.exists(segment_filepath) and os.path.getsize(segment_filepath) > 0:
                    duration = end_time - start_time
                    total_generation_time += duration
                    generated_files.append(segment_filepath)
                    success = True
                    yield {"type": "progress", "status": "segment_complete", "message": f"Segment {i}/{total_lines} OK ({duration:.2f}s).", "current": i, "total": total_lines}
                else:
                     error_message = f"File {segment_filename} empty/not created."
                     if os.path.exists(segment_filepath): await asyncio.to_thread(os.remove, segment_filepath)
            except Exception as e:
                 end_time = time.monotonic()
                 duration = end_time - start_time
                 error_message = f"Error generating {segment_filename} ({duration:.2f}s): {e}"
                 print(f"[generate.py] Error Detail: {error_message}") # Log error
                 if os.path.exists(segment_filepath): await asyncio.to_thread(os.remove, segment_filepath)

            if not success:
                has_errors = True
                yield {"type": "warning", "status": "segment_failed", "message": f"Segment {i}/{total_lines} failed.", "detail": error_message, "current": i, "total": total_lines}

        status_data = {"current": total_lines, "total": total_lines}
        yield {"type": "progress", "status": "generation_complete", "message": f"Generation phase finished. Successful: {len(generated_files)}/{total_lines}.", **status_data}
        if generated_files: print(f"[generate.py] Total async generation time: {total_generation_time:.2f}s")

        # --- Combine files ---
        pause_durations = None
        combination_needed = len(generated_files) > 0
        combination_successful = False # Track combination outcome

        if combination_needed:
             yield {"type": "progress", "status": "combination_start", "message": "Starting combination...", **status_data}
             if len(generated_files) > 1:
                 if len(generated_files) == total_lines:
                     pause_durations = [line["pause_after"] for line in conversation_data[:-1]]
                 else:
                     print("[generate.py] Warning: Combining without pauses due to skipped segments.")
                     pause_durations = None
                 # Run blocking combine_wav_files in thread
                 combination_successful = await asyncio.to_thread(combine_wav_files, generated_files, combined_filename, pause_durations=pause_durations)
             elif len(generated_files) == 1:
                 print("[generate.py] Copying single segment...")
                 try:
                     output_dir = os.path.dirname(os.path.abspath(combined_filename))
                     if output_dir: await asyncio.to_thread(os.makedirs, output_dir, exist_ok=True)
                     await asyncio.to_thread(shutil.copy2, generated_files[0], combined_filename)
                     combination_successful = True
                 except Exception as e: print(f"[generate.py] Error copying single segment: {e}")

             if combination_successful and os.path.exists(combined_filename):
                 final_combined_path = os.path.abspath(combined_filename)
                 yield {"type": "progress", "status": "combination_complete", "message": "Combination/copy successful.", **status_data}
             else:
                 has_errors = True # Mark overall process as having errors
                 yield {"type": "warning", "status": "combination_failed", "message": "Combination/copy phase failed.", **status_data}
        else:
             yield {"type": "progress", "status": "combination_skipped", "message": "Skipping combination (no files generated).", **status_data}


        # --- Clean up intermediate files ---
        if generated_files:
             yield {"type": "progress", "status": "cleanup_start", "message": "Cleaning up intermediate files...", **status_data}
             deleted_count = 0
             for f in generated_files:
                 try:
                     if await asyncio.to_thread(os.path.exists, f):
                         await asyncio.to_thread(os.remove, f)
                         deleted_count += 1
                 except Exception as e: print(f"[generate.py] Error deleting intermediate {f}: {e}")
             print(f"[generate.py] Intermediate cleanup: Deleted {deleted_count} files.")
             try:
                 if not await asyncio.to_thread(os.listdir, intermediate_dir):
                     await asyncio.to_thread(os.rmdir, intermediate_dir)
                     print(f"[generate.py] Removed empty intermediate directory.")
             except Exception as e: print(f"[generate.py] Error removing intermediate dir {intermediate_dir}: {e}")
             yield {"type": "progress", "status": "cleanup_complete", "message": "Cleanup complete.", **status_data}

    except Exception as e:
         print(f"[generate.py] Critical error during async processing: {e}", exc_info=True)
         yield {"type": "result", "status": "error", "message": f"Critical error during processing: {e}", "path": None}
         return

    # --- Yield final result ---
    final_status = "finished" if not has_errors and final_combined_path else "finished_with_errors" if final_combined_path else "error"
    final_message = "Processing finished."
    if has_errors: final_message = "Processing finished with errors."
    if not final_combined_path and combination_needed: final_message = "Processing failed (combination step)."

    yield {"type": "result", "status": final_status, "message": final_message, "path": final_combined_path}

# --- Main block for example ---
def main():
    """Example usage of the synchronous process_conversation function."""
    combined_filename = "combined_conversation_example.wav"
    intermediate_dir = "temp_audio_example"
    conversation = [
        {"text": "This is a short test.", "voice": "mia", "pause_after": 0.5},
        {"text": "Just to see if it works.", "voice": "zac", "pause_after": 0.0}
    ]
    final_file = process_conversation(conversation, combined_filename, intermediate_dir)
    if final_file: print(f"\n--- Example Complete: {final_file} ---")
    else: print("\n--- Example Failed ---")
    print("\nAvailable voices:")
    if AVAILABLE_VOICES: print("\n".join(f"- {v}" for v in AVAILABLE_VOICES))
    else: print("None found.")

if __name__ == "__main__":
    main() 