# # modules/stt/listener.py — Mic capture + Groq Whisper API (no local model needed)
# # Uses Groq's hosted Whisper — no PyTorch, no DLL issues, faster than local.

# import io
# import numpy as np
# import sounddevice as sd
# import soundfile as sf
# from groq import Groq

# from utils.logger import logger
# from config import GROQ_API_KEY, SILENCE_THRESHOLD

# # Force correct sample rate matching AMD mic's native rate
# SAMPLE_RATE = 44100
# MIC_DEVICE  = 1   # Microphone Array (AMD Audio Device)

# # Groq client — loads instantly, no heavy model download
# _client = Groq(api_key=GROQ_API_KEY)
# logger.info("Groq Whisper STT client ready ✓")


# def listen() -> str:
#     """
#     Record from microphone until silence detected.
#     Send audio to Groq Whisper API for transcription.
#     Handles Hindi / English / Hinglish automatically.

#     Returns:
#         Transcribed text string. Empty string if nothing heard.
#     """
#     logger.info("🎙️  Listening... (speak now)")

#     audio_chunks     = []
#     silence_chunks   = 0
#     chunk_duration   = 0.5
#     chunk_samples    = int(SAMPLE_RATE * chunk_duration)
#     max_silence      = max(int(SILENCE_THRESHOLD / chunk_duration), 3)  # at least 1.5s silence
#     started_speaking = False

#     try:
#         with sd.InputStream(
#             device=MIC_DEVICE,
#             samplerate=SAMPLE_RATE,
#             channels=1,
#             dtype='float32'
#         ) as stream:
#             while True:
#                 chunk, _ = stream.read(chunk_samples)
#                 chunk    = chunk.flatten()
#                 volume   = float(np.sqrt(np.mean(chunk ** 2)))

#                 is_speech = volume > 0.003

#                 if is_speech:
#                     started_speaking = True
#                     silence_chunks   = 0
#                     audio_chunks.append(chunk)
#                 else:
#                     if started_speaking:
#                         silence_chunks += 1
#                         audio_chunks.append(chunk)
#                         if silence_chunks >= max_silence:
#                             logger.debug("Silence detected — recording complete")
#                             break

#     except Exception as e:
#         logger.error(f"Microphone stream error: {e}")
#         return ""

#     if not audio_chunks:
#         logger.warning("No audio captured — you may not have spoken")
#         return ""

#     full_audio = np.concatenate(audio_chunks)

#     if len(full_audio) < SAMPLE_RATE * 0.5:
#         logger.warning("Recording too short — skipping")
#         return ""

#     # Resample from 44100 → 16000 for Whisper (smaller file, faster API call)
#     target_rate  = 16000
#     resample_ratio = target_rate / SAMPLE_RATE
#     target_length  = int(len(full_audio) * resample_ratio)
#     resampled      = np.interp(
#         np.linspace(0, len(full_audio) - 1, target_length),
#         np.arange(len(full_audio)),
#         full_audio
#     )

#     # Convert numpy float32 → WAV bytes in memory (no temp file needed)
#     wav_buffer = io.BytesIO()
#     sf.write(wav_buffer, resampled, target_rate, format='WAV', subtype='PCM_16')
#     wav_buffer.seek(0)

#     # Send to Groq Whisper API
#     logger.debug("Sending audio to Groq Whisper...")
#     try:
#         transcription = _client.audio.transcriptions.create(
#             file=("audio.wav", wav_buffer, "audio/wav"),
#             model="whisper-large-v3",
#             language=None,                # auto-detect Hindi/English/Hinglish
#             response_format="text"
#         )
#         transcript = transcription.strip() if transcription else ""
#         if transcript:
#             logger.info(f"Heard: '{transcript}'")
#         else:
#             logger.warning("Whisper returned empty transcript")
#         return transcript

#     except Exception as e:
#         logger.error(f"Groq Whisper API failed: {e}")
#         return ""



# modules/stt/listener.py — Mic capture + Groq Whisper API (no local model needed)
# Uses Groq's hosted Whisper — no PyTorch, no DLL issues, faster than local.

# modules/stt/listener.py — Mic capture + Groq Whisper API (no local model needed)
# Uses Groq's hosted Whisper — no PyTorch, no DLL issues, faster than local.

import io
import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq

from utils.logger import logger
from config import GROQ_API_KEY, SILENCE_THRESHOLD

# Force correct sample rate matching AMD mic's native rate
SAMPLE_RATE = 44100
MIC_DEVICE  = 1   # Microphone Array (AMD Audio Device)

# Groq client — loads instantly, no heavy model download
_client = Groq(api_key=GROQ_API_KEY)
logger.info("Groq Whisper STT client ready ✓")


def listen() -> str:
    """
    Record from microphone until silence detected.
    Send audio to Groq Whisper API for transcription.
    Handles Hindi / English / Hinglish automatically.
    """
    logger.info("🎙️  Listening... (speak now)")

    audio_chunks     = []
    silence_chunks   = 0
    chunk_duration   = 0.5
    chunk_samples    = int(SAMPLE_RATE * chunk_duration)
    max_silence      = max(int(SILENCE_THRESHOLD / chunk_duration), 3)
    started_speaking = False

    try:
        with sd.InputStream(
            device=MIC_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        ) as stream:
            while True:
                chunk, _ = stream.read(chunk_samples)
                chunk    = chunk.flatten()
                volume   = float(np.sqrt(np.mean(chunk ** 2)))

                is_speech = volume > 0.003

                if is_speech:
                    started_speaking = True
                    silence_chunks   = 0
                    audio_chunks.append(chunk)
                else:
                    if started_speaking:
                        silence_chunks += 1
                        audio_chunks.append(chunk)
                        if silence_chunks >= max_silence:
                            logger.debug("Silence detected — recording complete")
                            break

    except Exception as e:
        logger.error(f"Microphone stream error: {e}")
        return ""

    if not audio_chunks:
        logger.warning("No audio captured — you may not have spoken")
        return ""

    full_audio = np.concatenate(audio_chunks)

    if len(full_audio) < SAMPLE_RATE * 0.5:
        logger.warning("Recording too short — skipping")
        return ""

    return _transcribe_numpy(full_audio, SAMPLE_RATE)


def listen_from_file(path: str) -> str:
    """
    Transcribe a browser-recorded audio file (WebM/Opus or any format).
    Sends the raw file bytes directly to Groq Whisper —
    no soundfile needed, so WebM/Opus/MP4 all work fine.

    Args:
        path: Absolute path to the temp audio file saved by FastAPI.

    Returns:
        Transcribed text string. Empty string on failure.
    """
    logger.info(f"🎙️  Transcribing file: {path}")
    try:
        # Detect a sensible filename for Groq based on extension
        ext = path.rsplit('.', 1)[-1].lower() if '.' in path else 'webm'
        # Map common browser formats → Groq-accepted mime types
        mime_map = {
            'webm': 'audio/webm',
            'ogg':  'audio/ogg',
            'mp4':  'audio/mp4',
            'wav':  'audio/wav',
            'mp3':  'audio/mpeg',
        }
        mime = mime_map.get(ext, 'audio/webm')
        filename = f"recording.{ext}"

        with open(path, 'rb') as f:
            raw_bytes = f.read()

        if len(raw_bytes) < 1000:
            logger.warning("Audio file too small — likely empty recording")
            return ""

        logger.debug(f"Sending {len(raw_bytes)//1024} KB ({mime}) to Groq Whisper…")

        transcription = _client.audio.transcriptions.create(
            file=(filename, raw_bytes, mime),
            model="whisper-large-v3",
            language=None,          # auto-detect Hindi / English / Hinglish
            response_format="text"
        )

        transcript = transcription.strip() if transcription else ""
        if transcript:
            logger.info(f"Heard: '{transcript}'")
        else:
            logger.warning("Whisper returned empty transcript")
        return transcript

    except Exception as e:
        logger.error(f"listen_from_file error: {e}")
        return ""


# ══════════════════════════════════════════════════
# SHARED HELPER — resample numpy audio + send to Groq
# Used only by listen() (mic recording path)
# ══════════════════════════════════════════════════
def _transcribe_numpy(audio: np.ndarray, source_rate: int) -> str:
    target_rate = 16000

    if source_rate != target_rate:
        target_length = int(len(audio) * target_rate / source_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_length),
            np.arange(len(audio)),
            audio
        )

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, target_rate, format='WAV', subtype='PCM_16')
    wav_buffer.seek(0)

    logger.debug("Sending mic audio to Groq Whisper…")
    try:
        transcription = _client.audio.transcriptions.create(
            file=("audio.wav", wav_buffer, "audio/wav"),
            model="whisper-large-v3",
            language=None,
            response_format="text"
        )
        transcript = transcription.strip() if transcription else ""
        if transcript:
            logger.info(f"Heard: '{transcript}'")
        else:
            logger.warning("Whisper returned empty transcript")
        return transcript

    except Exception as e:
        logger.error(f"Groq Whisper API failed: {e}")
        return ""