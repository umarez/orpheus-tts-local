#!/usr/bin/env python3
"""
Simple example of using Orpheus TTS as a library.
This script demonstrates how to generate speech and save it to a file.
"""
import wave
import os
import time # Import the time module
from gguf_orpheus import generate_speech_from_api, AVAILABLE_VOICES

def text_to_speech(text, voice="tara", output_file=None):
    """
    Convert text to speech using Orpheus TTS.
    
    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (default: tara)
        output_file (str): Path to save the audio file (default: None)
    
    Returns:
        list: Audio segments
    """
    print(f"Converting: '{text}' with voice '{voice}'")
    
    # Generate speech
    audio_segments = generate_speech_from_api(
        prompt=text,
        voice=voice,
        output_file=output_file
    )
    
    return audio_segments

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
        return

    num_files = len(input_files)
    if pause_durations is not None and len(pause_durations) != num_files - 1:
        print(f"Error: Number of pause durations ({len(pause_durations)}) must be one less than the number of input files ({num_files}).")
        return

    # Ensure output directory exists if path includes directories
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    audio_data_list = []
    params = None
    valid_files_count = 0 # Keep track of files successfully read

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
                valid_files_count += 1 # Increment count of valid files
        except FileNotFoundError:
            print(f"Warning: File not found {filename}. Skipping.")
        except wave.Error as e:
            print(f"Warning: Error reading {filename}: {e}. Skipping.")

    if not audio_data_list or params is None:
        print("No valid audio data found or parameters could not be determined.")
        return

    # Adjust pause list if files were skipped
    if pause_durations is not None and valid_files_count != num_files:
         print(f"Warning: {num_files - valid_files_count} files were skipped due to errors. Pause durations may not align as intended.")
         # We proceed with the original pause list length, but it might be longer than needed.
         # A more robust solution might involve mapping pauses to successful reads, but that adds complexity.


    # Second pass: Write data and silence to output file
    try:
        with wave.open(output_file, 'wb') as wf:
            wf.setparams(params)
            nchannels, sampwidth, framerate, _, _, _ = params

            for i, audio_data in enumerate(audio_data_list):
                # Add pause *before* the current segment (if not the first)
                if i > 0 and pause_durations is not None:
                    # Ensure we don't go out of bounds for pauses if files were skipped
                    if i - 1 < len(pause_durations):
                        current_pause_duration = pause_durations[i-1]
                        num_pause_frames = int(current_pause_duration * framerate)
                        silence_bytes = b'\x00' * (num_pause_frames * nchannels * sampwidth)
                        wf.writeframes(silence_bytes)
                    else:
                         print(f"Warning: Skipping pause before segment {i+1} due to previous file errors.")


                wf.writeframes(audio_data) # Write the actual audio

        if pause_durations is not None:
             print(f"Successfully combined {valid_files_count} files with specified pauses into {output_file}")
        else:
             print(f"Successfully combined {valid_files_count} files (no pauses) into {output_file}")

    except wave.Error as e:
        print(f"Error writing combined WAV file {output_file}: {e}")
    except Exception as e:
         print(f"An unexpected error occurred while writing the combined file: {e}")

def main():
    # Define conversation with text, voice, filename, and pause AFTER this line
    conversation = [
    {"text": """Hey Leo, got a sec? I wanted to run something by you.""", "voice": "jess", "filename": "project_convo_jess_1.wav", "pause_after": 1.0},
    {"text": """Sure, Jess. What's up?""", "voice": "leo", "filename": "project_convo_leo_1.wav", "pause_after": 0.8},
    {"text": """It's about the new marketing campaign. I was thinking we could try a different approach for social media this time.""", "voice": "jess", "filename": "project_convo_jess_2.wav", "pause_after": 1.3},
    {"text": """Oh? Like what?""", "voice": "leo", "filename": "project_convo_leo_2.wav", "pause_after": 0.9},
    {"text": """Instead of just posting static images and links, maybe we could do more short video content. You know, behind-the-scenes glimpses, quick product demos, that kind of thing.""", "voice": "jess", "filename": "project_convo_jess_3.wav", "pause_after": 1.5},
    {"text": """Hmm, interesting. Video can be engaging, but it also takes more time and resources to produce.""", "voice": "leo", "filename": "project_convo_leo_3.wav", "pause_after": 1.2},
    {"text": """I know, but I think the potential reach and impact could be much higher. Plus, our competitors are starting to do more video, and we don't want to fall behind.""", "voice": "jess", "filename": "project_convo_jess_4.wav", "pause_after": 1.4},
    {"text": """That's a fair point. Have you thought about what kind of video content we could create?""", "voice": "leo", "filename": "project_convo_leo_4.wav", "pause_after": 1.1},
    {"text": """Yeah, I have a few ideas. We could do a series showcasing how our products are made, maybe interview some of our team members, or even create short tutorials on how to use our services.""", "voice": "jess", "filename": "project_convo_jess_5.wav", "pause_after": 1.6},
    {"text": """Those sound promising. What about the budget? Do we have enough allocated for video production?""", "voice": "leo", "filename": "project_convo_leo_5.wav", "pause_after": 1.3},
    {"text": """That's something we'd need to look into. I was hoping we could start small, maybe with just a few simple videos to test the waters and see how they perform.""", "voice": "jess", "filename": "project_convo_jess_6.wav", "pause_after": 1.1},
    {"text": """Okay, I'm open to exploring this further. Maybe we can schedule a quick meeting with the marketing team to brainstorm some concrete video ideas and discuss the budget implications?""", "voice": "leo", "filename": "project_convo_leo_6.wav", "pause_after": 1.4},
    {"text": """That would be great! How about tomorrow morning?""", "voice": "jess", "filename": "project_convo_jess_7.wav", "pause_after": 1.0},
    {"text": """Let me check my calendar... Yes, 10 am works for me.""", "voice": "leo", "filename": "project_convo_leo_7.wav", "pause_after": 1.2},
    {"text": """Perfect. I'll send out a meeting invite. Thanks for being open to this, Leo. I really think it could make a difference.""", "voice": "jess", "filename": "project_convo_jess_8.wav", "pause_after": 1.5},
    {"text": """No problem, Jess. It's good to try new things. Let's see what the team comes up with.""", "voice": "leo", "filename": "project_convo_leo_8.wav", "pause_after": 1.0},
    {"text": """Sounds like a plan!""", "voice": "jess", "filename": "project_convo_jess_9.wav", "pause_after": 0.0}
]
    generated_files = [] # Keep track of successfully generated files
    total_lines = len(conversation)
    total_generation_time = 0

    # Generate all the audio files
    print("Generating individual audio segments...")
    # Use enumerate to track progress (start=1 for user-friendly counting)
    for i, line in enumerate(conversation, start=1):
        print(f"\n--- Generating segment {i}/{total_lines} ---")
        start_time = time.monotonic() # Record start time
        try:
             # Generate speech (text_to_speech already prints "Converting...")
             text_to_speech(
                 text=line["text"],
                 voice=line["voice"],
                 output_file=line["filename"]
             )
             end_time = time.monotonic() # Record end time

             # Check if file was actually created before adding
             if os.path.exists(line["filename"]):
                 duration = end_time - start_time
                 total_generation_time += duration
                 print(f"-> Successfully generated {line['filename']} in {duration:.2f} seconds.")
                 generated_files.append(line["filename"])
             else:
                 print(f"-> Failed: File {line['filename']} was not created after generation attempt.")
        except Exception as e:
             end_time = time.monotonic()
             duration = end_time - start_time
             print(f"-> Error generating {line['filename']} after {duration:.2f} seconds: {e}")

    print(f"\n--- Generation Phase Complete ---")
    print(f"Successfully generated {len(generated_files)} out of {total_lines} segments.")
    if generated_files:
        print(f"Total generation time for successful segments: {total_generation_time:.2f} seconds.")
        avg_time = total_generation_time / len(generated_files) if generated_files else 0
        print(f"Average time per successful segment: {avg_time:.2f} seconds.")


    # Extract pause durations from the conversation data
    # We need N-1 pauses for N files
    pause_durations = [line["pause_after"] for line in conversation[:-1]]

    combined_filename = "combined_conversation_5.wav"

    # Combine all generated files with the specified pauses
    if generated_files: # Only combine if there are files to combine
        # Check if the number of generated files matches the expected number for pauses
        if len(generated_files) == len(conversation):
             # If all files were generated, use the extracted pauses
             print(f"\nCombining {len(generated_files)} files with specified pauses...")
             combine_wav_files(generated_files, combined_filename, pause_durations=pause_durations)
        elif len(generated_files) > 1:
             # If some files failed, we can't reliably use the extracted pauses.
             # Fallback to no pauses or a default pause? Let's use no pauses for safety.
             print("\nWarning: Not all audio segments were generated successfully.")
             print(f"Combining {len(generated_files)} generated segments without specific pauses between them.")
             combine_wav_files(generated_files, combined_filename, pause_durations=None) # Pass None for pauses
        else:
            # Only 0 or 1 file generated, combination doesn't make sense with pauses
             print("\nSkipping combination as not enough files were generated successfully.")
             combined_filename = None # Ensure cleanup doesn't run if combination failed


        # --- Clean up individual files if combination was attempted ---
        if combined_filename and os.path.exists(combined_filename): # Check if combined file was created
            print(f"\nCleaning up individual audio segments...")
            deleted_count = 0
            failed_count = 0
            for file_to_delete in generated_files: # Use the list of successfully generated files
                try:
                    os.remove(file_to_delete)
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting file {file_to_delete}: {e}")
                    failed_count += 1
            print(f"Cleanup complete. Deleted {deleted_count} files, {failed_count} failures.")
        elif generated_files:
             print(f"\nSkipping cleanup as combined file '{combined_filename}' was not created successfully.")
        # --- End of cleanup ---

    else:
        print("\nSkipping combination and cleanup as no files were generated successfully.")


    print("\nAll available voices:")
    for voice in AVAILABLE_VOICES:
        print(f"- {voice}")

if __name__ == "__main__":
    main() 