from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import logging
import os

# --- Configure Logging ---
# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'decoder.log')

# Set up basic configuration
# You might want to configure this more globally in your main app (api.py)
# For simplicity here, we configure it within the module.
# Use basicConfig only if logging hasn't been configured elsewhere yet.
try:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
except ValueError:
     # Avoid error if basicConfig was already called (e.g., in api.py)
     pass


# Get a logger instance for this module
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---


# --- Load SNAC Model ---
try:
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(snac_device)
    logger.info(f"SNAC model loaded successfully on device: {snac_device}")
except Exception as e:
    logger.exception("Failed to load SNAC model!")
    raise
# --- End Model Loading ---


def convert_to_audio(multiframe, count):
    if len(multiframe) < 7:
        logger.debug(f"convert_to_audio count {count}: Not enough tokens in multiframe ({len(multiframe)} < 7). Skipping.")
        return None
    
    num_frames = len(multiframe) // 7
    if num_frames == 0:
        logger.warning(f"convert_to_audio count {count}: Not enough tokens for a complete frame. Length: {len(multiframe)}")
        return None
    frame = multiframe[:num_frames*7]
    
    # Pre-allocate tensors for better efficiency
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
    codes_1 = torch.zeros(num_frames*2, dtype=torch.int32, device=snac_device)
    codes_2 = torch.zeros(num_frames*4, dtype=torch.int32, device=snac_device)

    # Fill tensors efficiently
    try:
        for j in range(num_frames):
            i = 7*j
            codes_0[j] = frame[i]
            
            codes_1[j*2] = frame[i+1]
            codes_1[j*2+1] = frame[i+4]
            
            codes_2[j*4] = frame[i+2]
            codes_2[j*4+1] = frame[i+3]
            codes_2[j*4+2] = frame[i+5]
            codes_2[j*4+3] = frame[i+6]
    except IndexError as e:
         logger.error(f"convert_to_audio count {count}: IndexError during tensor filling. Frame length: {len(frame)}, num_frames: {num_frames}, j={j}, i={i}. Error: {e}")
         return None

    # Debug log before validation
    logger.debug(f"convert_to_audio count {count}: Before validation:")
    logger.debug(f"  Input multiframe (len={len(multiframe)}): {multiframe[-21:]}")
    logger.debug(f"  codes_0 (shape={codes_0.shape}) min/max: {torch.min(codes_0)} / {torch.max(codes_0)}")
    logger.debug(f"  codes_1 (shape={codes_1.shape}) min/max: {torch.min(codes_1)} / {torch.max(codes_1)}")
    logger.debug(f"  codes_2 (shape={codes_2.shape}) min/max: {torch.min(codes_2)} / {torch.max(codes_2)}")

    # Validate token range in one operation
    if (torch.any(codes_0 < 0) or torch.any(codes_0 > 4096) or 
        torch.any(codes_1 < 0) or torch.any(codes_1 > 4096) or 
        torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)):
        logger.warning(f"convert_to_audio count {count}: Invalid token values detected!")
        if torch.any(codes_0 < 0) or torch.any(codes_0 > 4096): logger.warning(f"   codes_0 out of range [0, 4096]: min={torch.min(codes_0)}, max={torch.max(codes_0)}")
        if torch.any(codes_1 < 0) or torch.any(codes_1 > 4096): logger.warning(f"   codes_1 out of range [0, 4096]: min={torch.min(codes_1)}, max={torch.max(codes_1)}")
        if torch.any(codes_2 < 0) or torch.any(codes_2 > 4096): logger.warning(f"   codes_2 out of range [0, 4096]: min={torch.min(codes_2)}, max={torch.max(codes_2)}")
        return None

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    
    try:
        with torch.inference_mode():
            audio_hat = model.decode(codes)
    except Exception as e:
         logger.exception(f"convert_to_audio count {count}: Error during model.decode().")
         return None

    # Use a consistent slice size to avoid gaps
    if audio_hat.shape[2] >= 4096:
        audio_slice = audio_hat[:, :, 2048:4096]
    else:
         logger.warning(f"convert_to_audio count {count}: Audio output shape {audio_hat.shape} too small for standard slicing. Adjusting slice.")
         start_idx = max(0, audio_hat.shape[2] // 2 - 1024)
         end_idx = start_idx + 2048
         audio_slice = audio_hat[:, :, start_idx:min(end_idx, audio_hat.shape[2])]

    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes

def turn_token_into_id(token_string, index):
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        logger.debug(f"turn_token_into_id index {index}: No '<custom_token_' found in '{token_string}'")
        return None
    
    last_token = token_string[last_token_start:]
    
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            base_id = int(number_str)
            offset = (index % 7) * 4096
            final_id = base_id - 10 - offset
            return final_id
        except ValueError:
            logger.warning(f"turn_token_into_id index {index}: ValueError parsing number from '{last_token}'")
            return None
    else:
        logger.debug(f"turn_token_into_id index {index}: Token format mismatch for '{last_token}'")
        return None

async def tokens_decoder(token_gen):
    buffer = []
    count = 0
    overlap = 7
    process_threshold = 28

    async for token_sim in token_gen:
        logger.debug(f"tokens_decoder count {count}: Raw token_sim: '{token_sim}'")
        token = turn_token_into_id(token_sim, count)
        logger.debug(f"tokens_decoder count {count}: Parsed token: {token}")
        if token is not None and token >= 0:
            buffer.append(token)
            count += 1

            if count % 7 == 0 and count >= process_threshold:
                chunk_size = 28
                if len(buffer) >= chunk_size:
                     buffer_to_proc = buffer[-chunk_size:]
                     audio_samples = convert_to_audio(buffer_to_proc, count)
                     if audio_samples:
                         yield audio_samples

# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    audio_queue = queue.Queue(maxsize=20)

    async def async_token_gen_wrapper():
        logger.debug("tokens_decoder_sync: Starting async_token_gen_wrapper")
        for token in syn_token_gen:
            yield token
        logger.debug("tokens_decoder_sync: Finished async_token_gen_wrapper")

    async def async_producer():
        logger.debug("tokens_decoder_sync: Starting async_producer")
        try:
            async for audio_chunk in tokens_decoder(async_token_gen_wrapper()):
                if audio_chunk:
                    try:
                        audio_queue.put(audio_chunk, timeout=2.0)
                    except queue.Full:
                        logger.warning("tokens_decoder_sync: Audio queue full. Dropping oldest chunk.")
                        try:
                            audio_queue.get_nowait()
                            audio_queue.put(audio_chunk, timeout=0.1)
                        except Exception:
                             logger.error("tokens_decoder_sync: Failed to manage full queue.")
                             pass
            logger.debug("tokens_decoder_sync: tokens_decoder finished.")
        except Exception as e:
            logger.exception(f"tokens_decoder_sync: Error in async_producer.")
        finally:
            logger.debug("tokens_decoder_sync: Putting sentinel None in queue.")
            audio_queue.put(None)

    def run_async_in_thread():
        logger.debug("tokens_decoder_sync: Starting background thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_producer())
        finally:
            loop.close()
            logger.debug("tokens_decoder_sync: Background thread event loop closed.")

    thread = threading.Thread(target=run_async_in_thread, name="AudioProducerThread")
    thread.daemon = True
    thread.start()

    last_audio = None
    logger.debug("tokens_decoder_sync: Starting to yield audio from queue.")
    while True:
        try:
            audio = audio_queue.get(timeout=10.0)
            if audio is None:
                logger.debug("tokens_decoder_sync: Received sentinel None. Exiting.")
                break
            last_audio = audio
            yield audio
        except queue.Empty:
            logger.warning("tokens_decoder_sync: Audio queue empty after timeout. Repeating last audio chunk.")
            if last_audio is not None:
                 yield last_audio
            else:
                 logger.warning("tokens_decoder_sync: Queue empty and no previous audio. Breaking loop.")
                 break

    logger.debug("tokens_decoder_sync: Waiting for background thread to join.")
    thread.join(timeout=1.0)
    if thread.is_alive():
         logger.warning("tokens_decoder_sync: Background thread did not exit cleanly.")
    else:
         logger.debug("tokens_decoder_sync: Background thread joined successfully.")