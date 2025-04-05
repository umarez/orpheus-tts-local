#!/usr/bin/env python3
"""
Reusable functions for using Orpheus TTS and example usage.
"""
import wave
import os
import time
import sys

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

def text_to_speech(text, voice="tara", output_file=None):
    """
    Convert text to speech using Orpheus TTS.

    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (default: tara)
        output_file (str): Path to save the audio file (default: None)

    Returns:
        list: Audio segments (or path if output_file is specified - depends on underlying library)
    """
    print(f"Converting: '{text}' with voice '{voice}'")

    # Generate speech
    # Assuming generate_speech_from_api handles file writing if output_file is given
    # and might return segments or status. Adjust return based on actual library behavior.
    result = generate_speech_from_api(
        prompt=text,
        voice=voice,
        output_file=output_file
    )

    # Return value might need adjustment based on what generate_speech_from_api returns
    return result # Or maybe return output_file if it exists?

def combine_wav_files(input_files, output_file, pause_durations=None):
    """
    Combines multiple WAV files into a single WAV file with specified pauses in between.

    Args:
        input_files (list): A list of paths to the input WAV files in order.
        output_file (str): The path to save the combined WAV file.
        pause_durations (list, optional): A list of pause durations in seconds
                                         for the gaps between files. Length should be
                                         len(input_files) - 1. If None, no pauses added.
    """
    if not input_files:
        print("No input files provided for combining.")
        return False # Indicate failure

    num_files = len(input_files)
    if pause_durations is not None and len(pause_durations) != num_files - 1:
        print(f"Error: Number of pause durations ({len(pause_durations)}) must be one less than the number of input files ({num_files}).")
        return False # Indicate failure

    # Ensure output directory exists if path includes directories
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    audio_data_list = []
    params = None
    valid_files_count = 0 # Keep track of files successfully read
    actual_input_files_used = [] # Track which files were actually read

    # First pass: Read all valid files and get parameters
    for filename in input_files:
        try:
            with wave.open(filename, 'rb') as wf:
                current_params = wf.getparams()
                if params is None:
                    params = current_params
                # Check essential parameters: nchannels, sampwidth, framerate
                elif params[:3] != current_params[:3]:
                     print(f"Warning: Skipping {filename}. Incompatible parameters (channels, width, or rate).")
                     continue # Skip this file

                audio_data_list.append(wf.readframes(wf.getnframes()))
                actual_input_files_used.append(filename) # Add to list of used files
                valid_files_count += 1 # Increment count of valid files
        except FileNotFoundError:
            print(f"Warning: File not found {filename}. Skipping.")
        except wave.Error as e:
            print(f"Warning: Error reading {filename}: {e}. Skipping.")

    if not audio_data_list or params is None:
        print("No valid audio data found or parameters could not be determined.")
        return False # Indicate failure

    # Adjust pause list based on *successfully read* files
    adjusted_pause_durations = None
    if pause_durations is not None:
        if valid_files_count != num_files:
            print(f"Warning: {num_files - valid_files_count} files were skipped. Adjusting pauses.")
            # This is tricky. A simple approach is to assume pauses correspond to the *original* list
            # and only apply pauses between *successfully read consecutive* files from the original list.
            # For now, let's just use the original list length, but acknowledge it might be wrong.
            # A better but more complex way would be to map original indices to successful read indices.
            if len(pause_durations) >= valid_files_count -1:
                 adjusted_pause_durations = pause_durations[:valid_files_count - 1]
            else:
                 # This case shouldn't happen if the initial length check passed, but safety first
                 print(f"Warning: Mismatch in pause durations after skipping files. Applying available pauses.")
                 adjusted_pause_durations = pause_durations
        else:
             adjusted_pause_durations = pause_durations


    # Second pass: Write data and silence to output file
    try:
        with wave.open(output_file, 'wb') as wf:
            wf.setparams(params)
            nchannels, sampwidth, framerate, _, _, _ = params

            for i, audio_data in enumerate(audio_data_list):
                # Add pause *before* the current segment (if not the first)
                if i > 0 and adjusted_pause_durations is not None:
                    # Use the adjusted list index
                    if i - 1 < len(adjusted_pause_durations):
                        current_pause_duration = adjusted_pause_durations[i-1]
                        if current_pause_duration > 0:
                            num_pause_frames = int(current_pause_duration * framerate)
                            silence_bytes = b'\x00' * (num_pause_frames * nchannels * sampwidth)
                            wf.writeframes(silence_bytes)
                    # No warning here, as pauses are intentionally adjusted


                wf.writeframes(audio_data) # Write the actual audio

        if adjusted_pause_durations is not None:
             print(f"Successfully combined {valid_files_count} files with adjusted pauses into {output_file}")
        else:
             print(f"Successfully combined {valid_files_count} files (no pauses) into {output_file}")
        return True # Indicate success

    except wave.Error as e:
        print(f"Error writing combined WAV file {output_file}: {e}")
        return False # Indicate failure
    except Exception as e:
         print(f"An unexpected error occurred while writing the combined file: {e}")
         return False # Indicate failure


def process_conversation(conversation_data, combined_filename, intermediate_dir="temp_audio"):
    """
    Generates individual audio files for a conversation, combines them,
    and cleans up intermediate files.

    Args:
        conversation_data (list): List of dictionaries, each defining a line
                                   (text, voice, pause_after).
        combined_filename (str): Path to save the final combined WAV file.
        intermediate_dir (str): Directory to store temporary segment files.

    Returns:
        str: Absolute path to the generated combined file if successful, None otherwise.
    """
    os.makedirs(intermediate_dir, exist_ok=True)
    generated_files = [] # Keep track of successfully generated file paths
    filenames_only = [] # Keep track of basenames for logging
    total_lines = len(conversation_data)
    total_generation_time = 0

    # Generate all the audio files
    print("Generating individual audio segments...")
    for i, line in enumerate(conversation_data, start=1):
        print(f"\n--- Generating segment {i}/{total_lines} ---")
        # Create a unique filename for the intermediate file
        segment_filename = f"segment_{i}_{line.get('voice', 'default')}_{int(time.time() * 1000)}.wav"
        segment_filepath = os.path.join(intermediate_dir, segment_filename)

        start_time = time.monotonic() # Record start time
        try:
             # Generate speech directly to the intermediate file
             text_to_speech(
                 text=line["text"],
                 voice=line["voice"],
                 output_file=segment_filepath # Save directly
             )
             end_time = time.monotonic() # Record end time

             # Check if file was actually created before adding
             if os.path.exists(segment_filepath) and os.path.getsize(segment_filepath) > 0: # Check size > 0 too
                 duration = end_time - start_time
                 total_generation_time += duration
                 print(f"-> Successfully generated {segment_filename} in {duration:.2f} seconds.")
                 generated_files.append(segment_filepath) # Store full path
                 filenames_only.append(segment_filename)
             else:
                 print(f"-> Failed: File {segment_filename} was not created or is empty.")
                 # Attempt cleanup if failed
                 if os.path.exists(segment_filepath):
                     try:
                         os.remove(segment_filepath)
                     except OSError:
                         pass # Ignore cleanup error if file didn't exist anyway
        except Exception as e:
             end_time = time.monotonic()
             duration = end_time - start_time
             print(f"-> Error generating {segment_filename} after {duration:.2f} seconds: {e}")
             # Attempt cleanup on error
             if os.path.exists(segment_filepath):
                 try:
                     os.remove(segment_filepath)
                 except OSError:
                     pass

    print(f"\n--- Generation Phase Complete ---")
    print(f"Successfully generated {len(generated_files)} out of {total_lines} segments.")
    if generated_files:
        print(f"Generated files: {', '.join(filenames_only)}")
        print(f"Total generation time for successful segments: {total_generation_time:.2f} seconds.")
        avg_time = total_generation_time / len(generated_files) if generated_files else 0
        print(f"Average time per successful segment: {avg_time:.2f} seconds.")

    # Extract pause durations from the conversation data *corresponding to generated files*
    # This requires mapping generated_files back to conversation_data, which is complex if failures occurred.
    # Simpler: Use pauses from the original list corresponding to the *number* of successfully generated files.
    pause_durations = None
    if len(generated_files) > 1:
        # Check if the number of generated files allows using original pauses
        if len(generated_files) == total_lines:
             pause_durations = [line["pause_after"] for line in conversation_data[:-1]]
        else:
             print("\nWarning: Not all segments generated. Pause timing might be inaccurate.")
             # Attempt to use pauses corresponding to the first N-1 generated segments
             # This assumes the *first* N segments were the ones generated successfully. Risky assumption.
             # A safer fallback is no pauses, or using pauses only if len == total_lines
             pause_durations = None # Fallback to no pauses for safety if segments missing
             # Alternatively, could try: pause_durations = [line["pause_after"] for line in conversation_data[:len(generated_files)-1]]


    final_combined_path = None # Track the final file path

    # Combine all generated files with the specified pauses
    if len(generated_files) > 1: # Only combine if there are multiple files
        print(f"\nCombining {len(generated_files)} successfully generated files...")
        combination_successful = combine_wav_files(
            generated_files,
            combined_filename,
            pause_durations=pause_durations
        )

        if combination_successful and os.path.exists(combined_filename):
             final_combined_path = os.path.abspath(combined_filename) # Store the absolute path
             print(f"Combination successful: {final_combined_path}")
        else:
             print(f"Combination failed or output file '{combined_filename}' not found.")
             final_combined_path = None # Ensure it's None if combination failed

    elif len(generated_files) == 1:
         print("\nOnly one segment generated. Copying it as the final file.")
         try:
             # Ensure output directory exists
             output_dir = os.path.dirname(os.path.abspath(combined_filename))
             if output_dir:
                 os.makedirs(output_dir, exist_ok=True)
             shutil.copy2(generated_files[0], combined_filename)
             final_combined_path = os.path.abspath(combined_filename)
             print(f"Copied single segment to {final_combined_path}")
         except Exception as e:
             print(f"Error copying single segment file: {e}")
             final_combined_path = None
    else:
         print("\nSkipping combination as no files were generated successfully.")


    # --- Clean up intermediate files ---
    if generated_files:
        print(f"\nCleaning up {len(generated_files)} intermediate audio segments...")
        deleted_count = 0
        failed_count = 0
        for file_to_delete in generated_files: # Use the list of successfully generated files
            try:
                if os.path.exists(file_to_delete): # Check again before deleting
                     os.remove(file_to_delete)
                     deleted_count += 1
                else:
                     print(f"Intermediate file already removed or not found: {file_to_delete}")
            except OSError as e:
                print(f"Error deleting intermediate file {file_to_delete}: {e}")
                failed_count += 1
        print(f"Intermediate cleanup complete. Deleted {deleted_count} files, {failed_count} failures.")
        # Optionally remove the intermediate directory if empty
        try:
             if not os.listdir(intermediate_dir):
                 os.rmdir(intermediate_dir)
                 print(f"Removed empty intermediate directory: {intermediate_dir}")
        except OSError as e:
             print(f"Could not remove intermediate directory {intermediate_dir}: {e}")

    # --- End of cleanup ---


    return final_combined_path


def main():
    """Example usage of the process_conversation function."""
    import shutil # Needed for single file copy logic

    combined_filename = "combined_conversation_example.wav"
    intermediate_dir = "temp_audio_example" # Use a specific dir for example

    # Define conversation (no filenames needed here anymore)
    conversation = [
        # {"text": """Hey Zac, did you manage to book the conference room for our presentation rehearsal?""", "voice": "mia", "pause_after": 1.1},
        # {"text": """Hi Mia! Yeah, I did. We've got it for tomorrow afternoon, from 2 to 4.""", "voice": "zac", "pause_after": 1.0},
        # {"text": """Perfect! That gives us plenty of time to run through everything a couple of times. <sigh> I'm still a bit nervous about the Q&A section.""", "voice": "mia", "pause_after": 1.4},
        # {"text": """Don't worry, Mia. We've prepared well. And we can definitely practice some potential questions during the rehearsal.""", "voice": "zac", "pause_after": 1.2},
        # {"text": """That would be a huge help. I always feel like my mind goes blank under pressure. <groan>""", "voice": "mia", "pause_after": 1.3},
        # {"text": """Just take a deep breath. We can even write down some key talking points to refer to if needed. Think of it as a safety net.""", "voice": "zac", "filename": "rehearsal_convo_zac_3.wav", "pause_after": 1.1},
        # {"text": """That's a good idea. <chuckle> Safety net for my brain. I like it. Did you finalize the slides with the updated market research data?""", "voice": "mia", "filename": "rehearsal_convo_mia_4.wav", "pause_after": 1.5},
        # {"text": """Yep, all updated and double-checked. I also added that extra graph we discussed. It really strengthens our argument.""", "voice": "zac", "filename": "rehearsal_convo_zac_4.wav", "pause_after": 1.3},
        # {"text": """Fantastic! You're a lifesaver, Zac. <gasp> Oh, and did you remember to bring the clicker?""", "voice": "mia", "filename": "rehearsal_convo_mia_5.wav", "pause_after": 1.0},
        # {"text": """<laugh> Of course! Wouldn't want you doing the awkward reach-across-the-laptop move. I've got it right here.""", "voice": "zac", "filename": "rehearsal_convo_zac_5.wav", "pause_after": 1.2},
        # {"text": """You know me too well. Okay, so tomorrow at 2 in the conference room. Anything else we need to prepare beforehand?""", "voice": "mia", "filename": "rehearsal_convo_mia_6.wav", "pause_after": 1.4},
        # {"text": """Hmm, maybe just review your notes one last time tonight. And get a good night's sleep! We want to be sharp tomorrow.""", "voice": "zac", "filename": "rehearsal_convo_zac_6.wav", "pause_after": 1.1},
        # {"text": """Good advice. I was planning on doing that anyway. <yawn> This week has been exhausting.""", "voice": "mia", "filename": "rehearsal_convo_mia_7.wav", "pause_after": 1.0},
        # {"text": """Tell me about it. But we're almost there. Just this presentation to nail, and then we can all relax a bit.""", "voice": "zac", "filename": "rehearsal_convo_zac_7.wav", "pause_after": 1.3},
        # {"text": """That's what I'm looking forward to. Okay, Zac. Thanks for booking the room and for all your help.""", "voice": "mia", "filename": "rehearsal_convo_mia_8.wav", "pause_after": 1.1},
        # {"text": """No problem, Mia. We're a team. See you tomorrow!""", "voice": "zac", "pause_after": 0.0} # Last line pause is ignored by combine logic
    ]

    # Call the processing function
    final_file = process_conversation(conversation, combined_filename, intermediate_dir)

    if final_file:
        print(f"\n--- Example Complete ---")
        print(f"Successfully generated conversation: {final_file}")
    else:
        print("\n--- Example Failed ---")
        print("Conversation generation failed.")


    print("\nAvailable voices (from gguf_orpheus):")
    if AVAILABLE_VOICES:
        for voice in AVAILABLE_VOICES:
            print(f"- {voice}")
    else:
        print("No voices found.")

if __name__ == "__main__":
    main() 