# tts/speaker.py — Text-to-Speech wrapper.
# Supports gTTS (prototype, free) and ElevenLabs (production, best voice quality).

import os
import threading
import tempfile
from utils.logger import logger
from config import (
    TTS_ENGINE, TTS_LANGUAGE, TTS_SLOW,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
)

# ── Global lock — only ONE speak() runs at a time across ALL threads ──
_tts_lock = threading.Lock()


class Speaker:
    """
    Speaks text aloud using the configured TTS engine.
    Thread-safe: uses a global lock so detection thread and main pipeline
    never speak simultaneously.
    Usage: Speaker().speak("Hello, you are in a café.")
    """

    def speak(self, text: str):
        """Convert text to speech and play it immediately."""
        if not text or not text.strip():
            logger.warning("Speaker received empty text — skipping")
            return

        preview = text[:70] + "..." if len(text) > 70 else text
        logger.info(f"🔊 Speaking: '{preview}'")

        # ✅ Acquire lock — if another thread is speaking, wait for it to finish
        with _tts_lock:
            if TTS_ENGINE == "elevenlabs":
                self._speak_elevenlabs(text)
            else:
                self._speak_gtts(text)

    # ── gTTS (prototype) ──────────────────────────────
    def _speak_gtts(self, text: str):
        try:
            from gtts import gTTS
            import playsound

            tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name

            tts.save(temp_path)
            playsound.playsound(temp_path)

        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            print(f"\n[SPEECH OUTPUT]: {text}\n")

        finally:
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass

    # ── ElevenLabs (production) ───────────────────────
    def _speak_elevenlabs(self, text: str):
        try:
            from elevenlabs import ElevenLabs, play

            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            audio = client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                text=text,
                model_id="eleven_turbo_v2",
            )
            play(audio)

        except Exception as e:
            logger.error(f"ElevenLabs failed: {e} — falling back to gTTS")
            self._speak_gtts(text)