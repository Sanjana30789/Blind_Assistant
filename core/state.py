# core/state.py — The shared state object that flows through every LangGraph node.
# Every node reads from this and writes back to it.

from typing import TypedDict, Optional


class AssistantState(TypedDict):
    """
    Single source of truth for one complete request cycle:
    Voice → STT → Agent → Module → TTS
    """

    # ── Input stage ───────────────────────────────────
    raw_transcript:         str     # Raw Whisper output (messy, unfiltered)
    cleaned_transcript:     str     # What the agent understood the user meant

    # ── Agent decision ────────────────────────────────
    mode:                   str     # "navigation_mode" | "reading_mode" | "currency_mode" | "unknown"
    confidence:             float   # 0.0 → 1.0  (how sure the agent is)
    extra_context:          str     # e.g. "left side", "near door", "the sign ahead"

    # ── Module output ─────────────────────────────────
    raw_output:             str     # Direct VLM response before filtering
    final_output:           str     # Clean text, ready to be spoken

    # ── Control flags ─────────────────────────────────
    needs_clarification:    bool    # True = low confidence, must ask user first
    clarification_question: str     # The yes/no question to ask
    error:          Optional[str]   # Error message to speak if something fails
    retry_count:            int     # How many retries have happened this cycle