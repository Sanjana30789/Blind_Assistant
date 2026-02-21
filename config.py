# config.py — Central configuration. Change values here, applies everywhere.

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")

# ── STT ───────────────────────────────────────────────
WHISPER_MODEL       = "small"      # tiny / base / small / medium
WHISPER_DEVICE      = "cpu"        # cpu or cuda
SAMPLE_RATE         = 44100
SILENCE_THRESHOLD   = 2.0          # seconds of silence = user stopped speaking

# ── Agent (Groq LLM for routing) ──────────────────────
AGENT_MODEL         = "llama-3.1-8b-instant"   # fast + free on Groq
AGENT_TEMPERATURE   = 0.1

# ── Confidence Thresholds ─────────────────────────────
CONFIDENCE_HIGH     = 0.75         # act directly
CONFIDENCE_MEDIUM   = 0.50         # act but confirm with user
# below 0.50 = ask user one yes/no question

# ── Vision / VLM (Groq vision model) ─────────────────
VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq vision model
VLM_MAX_TOKENS      = 200
CAMERA_INDEX        = 0
CAMERA_WARMUP_MS    = 500

# ── TTS ───────────────────────────────────────────────
TTS_ENGINE          = "gtts"       # "gtts" or "elevenlabs"
TTS_LANGUAGE        = "en"
TTS_SLOW            = False