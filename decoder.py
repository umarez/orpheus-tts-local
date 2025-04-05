from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue

# Load model once
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(snac_device)

def convert_to_audio(multiframe, count):
    if len(multiframe) < 7:
        return None
    
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    
    # Pre-allocate tensors for better efficiency
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
    codes_1 = torch.zeros(num_frames*2, dtype=torch.int32, device=snac_device)
    codes_2 = torch.zeros(num_frames*4, dtype=torch.int32, device=snac_device)

    # Fill tensors efficiently
    for j in range(num_frames):
        i = 7*j
        codes_0[j] = frame[i]
        
        codes_1[j*2] = frame[i+1]
        codes_1[j*2+1] = frame[i+4]
        
        codes_2[j*4] = frame[i+2]
        codes_2[j*4+1] = frame[i+3]
        codes_2[j*4+2] = frame[i+5]
        codes_2[j*4+3] = frame[i+6]

    # Validate token range in one operation
    if (torch.any(codes_0 < 0) or torch.any(codes_0 > 4096) or 
        torch.any(codes_1 < 0) or torch.any(codes_1 > 4096) or 
        torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)):
        print(f"Invalid token values at count {count}")  # Add logging
        return None

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    
    with torch.inference_mode():
        audio_hat = model.decode(codes)
    
    # Use a consistent slice size to avoid gaps
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes

def turn_token_into_id(token_string, index):
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None

async def tokens_decoder(token_gen):
    buffer = []
    count = 0
    overlap = 7  # Ensure some overlap between chunks for smooth transitions
    
    async for token_sim in token_gen:       
        token = turn_token_into_id(token_sim, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Process when we have enough tokens, with overlap to ensure continuity
            if count % 7 == 0 and count > 27:
                # Use more of the buffer to ensure continuity
                buffer_to_proc = buffer[-(28 + overlap):]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    audio_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues

    # Convert the synchronous token generator into an async generator
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        try:
            async for audio_chunk in tokens_decoder(async_token_gen()):
                # Use a timeout to prevent blocking indefinitely
                try:
                    audio_queue.put(audio_chunk, timeout=2.0)
                except queue.Full:
                    # If queue is full, drop oldest item and add new one
                    try:
                        audio_queue.get_nowait()
                        audio_queue.put(audio_chunk)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"Error in audio producer: {e}")
        finally:
            audio_queue.put(None)  # Sentinel

    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_producer())
        loop.close()

    thread = threading.Thread(target=run_async)
    thread.daemon = True  # Allow program to exit even if thread is running
    thread.start()

    last_audio = None
    while True:
        try:
            audio = audio_queue.get(timeout=5.0)
            if audio is None:
                break
            last_audio = audio
            yield audio
        except queue.Empty:
            # If queue is empty for too long, repeat last audio to avoid complete silence
            if last_audio is not None:
                yield last_audio
            continue

    thread.join(0.5)  # Give thread a chance to clean up, but don't wait forever