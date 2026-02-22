# main.py — Single entry point. Run this: python main.py

import sys
import os
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["QWEN_2_5_ENABLED"] = "False"
os.environ["QWEN_3_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["SMOLVLM2_ENABLED"] = "False"
os.environ["DEPTH_ESTIMATION_ENABLED"] = "False"

# ── Path setup — must be FIRST before any local imports ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Create logs/ before logger initialises
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# ── Now safe to import local modules ──────────────────
from utils.logger import logger
from utils.audio_utils import check_microphone_available
from tts.speaker import Speaker
from modules.stt.listener import listen
from core.agent import agent
from core.state import AssistantState

speaker = Speaker()

# Optional memory placeholder
memory = {
    "last_scene": None
}


# ═══════════════════════════════════════════════
# SAFE STATE BUILDER
# ═══════════════════════════════════════════════
def build_state(transcript: str) -> AssistantState:
    """Always returns a fully-populated state so LangGraph never hits missing keys."""
    return {
        "raw_transcript":         transcript.strip(),
        "cleaned_transcript":     "",
        "mode":                   "unknown",      # safe default (not empty string)
        "confidence":             0.0,
        "extra_context":          "",
        "raw_output":             {},
        "final_output":           "",
        "needs_clarification":    False,
        "clarification_question": "",
        "error":                  None,
        "retry_count":            0,
        "spoken": False,
    }


# ═══════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════
def run_pipeline(transcript: str):
    if not transcript.strip():
        logger.warning("Skipping empty transcript")
        return

    logger.info(f"─── New request: '{transcript}' ───")

    state = build_state(transcript)

    try:
        # Invoke LangGraph agent
        result_state = agent.invoke(state)

        # Full debug dump — helps catch any future key errors
        logger.debug(f"Graph result keys: {list(result_state.keys())}")
        logger.debug(f"mode={result_state.get('mode')} | "
                     f"confidence={result_state.get('confidence')} | "
                     f"final_output={result_state.get('final_output', '')[:80]}")

        # Memory store
        if isinstance(result_state.get("raw_output"), dict):
            memory["last_scene"] = result_state["raw_output"]

        logger.info("─── Request complete ───")

    except KeyError as e:
        # Catches LangGraph state key mismatches — tells you EXACTLY what key is missing
        logger.error(f"Pipeline state key error — missing key: {e}", exc_info=True)
        speaker.speak("Sorry, I had trouble understanding that.")

    except ValueError as e:
        # Catches JSON / type conversion errors from agent nodes
        logger.error(f"Pipeline value error: {e}", exc_info=True)
        speaker.speak("Sorry, I had trouble understanding that.")

    except Exception as e:
        # Catch-all with full traceback
        logger.error(f"Pipeline error — type={type(e).__name__} | detail={e}", exc_info=True)
        speaker.speak("Sorry, I had trouble understanding that.")


# ═══════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════
def main():
    logger.info("══════════════════════════════════════")
    logger.info("    Blind Assistant — Starting Up     ")
    logger.info("══════════════════════════════════════")

    if not check_microphone_available():
        logger.error("No microphone found. Connect a microphone and restart.")
        sys.exit(1)

    speaker.speak("Assistant is ready. You can speak now.")
    logger.info("Listening loop started. Press Ctrl+C to quit.")

    while True:
        try:
            transcript = listen()

            if transcript:
                run_pipeline(transcript)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
            speaker.speak("Goodbye!")
            break

        except Exception as e:
            logger.error(f"Unhandled error in main loop — type={type(e).__name__} | {e}",
                         exc_info=True)
            speaker.speak("Something went wrong. Please try again.")


if __name__ == "__main__":
    main()