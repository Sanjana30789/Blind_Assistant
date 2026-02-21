# modules/knowledge/knowledge_logic.py

from utils.logger import logger
from tts.speaker import Speaker
from modules.knowledge.knowledge_tool import search_web
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import AGENT_MODEL, GROQ_API_KEY
from langdetect import detect
import datetime
import requests


llm = ChatGroq(
    model=AGENT_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2
)


# ─────────────────────────────────────────────
# Language Detection
# ─────────────────────────────────────────────
def _detect_language(query: str) -> str:
    try:
        code = detect(query)
        logger.info(f"Detected language code: {code}")
        return code  # 'hi', 'en', 'mr', 'ta', 'te', 'bn', 'gu', 'ur' etc.
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"


# ─────────────────────────────────────────────
# Local Data Fetchers
# ─────────────────────────────────────────────
def _get_current_time() -> str:
    return datetime.datetime.now().strftime("%I:%M %p")


def _get_current_date() -> str:
    return datetime.datetime.now().strftime("%A, %d %B %Y")


def _get_weather() -> str:
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=19.2183&longitude=73.0868&current_weather=true"
        data = requests.get(url, timeout=5).json()
        temp = data["current_weather"]["temperature"]
        return f"{temp} degrees Celsius"
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return "unavailable"


# ─────────────────────────────────────────────
# Single LLM Call — handles everything
# ─────────────────────────────────────────────
def _ask_llm(query: str, lang_code: str, web_context: str = "") -> str:
    now_time   = _get_current_time()
    today_date = _get_current_date()
    temperature = _get_weather()

    context_section = (
        f"Additional web context: {web_context}"
        if web_context else
        "No web context available — use your own knowledge."
    )

    prompt = f"""You are a helpful voice assistant for visually impaired users in India.

LANGUAGE RULE: The user is speaking in language with ISO code '{lang_code}'.
You MUST reply in that exact language only. Never switch to any other language under any circumstance.

LIVE DATA AVAILABLE (always use these if the question is about time, date, or weather):
- Current time: {now_time}
- Today's date: {today_date}
- Current temperature (Dombivli): {temperature}

{context_section}

Rules:
- Answer in MAXIMUM 2 short, natural sentences.
- Give the direct answer immediately — no preamble like "Here is", "According to", "Sure".
- Speak naturally as if talking to a person out loud.
- If you don't know something, say so honestly in language code '{lang_code}'.

Question: {query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().replace("\n", " ")


# ─────────────────────────────────────────────
# Main Handler
# ─────────────────────────────────────────────
def handle_knowledge_query(query: str):
    try:
        logger.info(f"Knowledge query: {query}")

        # Detect language upfront
        lang_code = _detect_language(query)

        # Optional web boost — never block on failure
        web_context = ""
        try:
            web_context = search_web(query)
            if web_context:
                logger.info("Web search succeeded — using as context boost")
            else:
                logger.warning("Web search empty — LLM will use own knowledge")
        except Exception as e:
            logger.warning(f"Web search skipped: {e}")

        # Single LLM call handles everything
        answer = _ask_llm(query, lang_code, web_context)
        Speaker().speak(answer)

    except Exception as e:
        logger.error(f"Knowledge logic error: {e}", exc_info=True)
        try:
            lang_code = _detect_language(query)
            answer = _ask_llm(
                query,
                lang_code,
                "An error occurred. Apologize briefly in the user's language."
            )
            Speaker().speak(answer)
        except Exception:
            Speaker().speak("Sorry, I couldn't process that.")