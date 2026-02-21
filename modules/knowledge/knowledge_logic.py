# modules/knowledge/knowledge_logic.py

from utils.logger import logger
from tts.speaker import Speaker
from modules.knowledge.knowledge_tool import search_web
from langchain_groq import ChatGroq
from config import AGENT_MODEL, GROQ_API_KEY
import datetime
import requests


llm = ChatGroq(
    model=AGENT_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2
)


def handle_knowledge_query(query: str):
    try:
        logger.info(f"Knowledge query: {query}")

        query_lower = query.lower()

        # ─────────────────────────────
        # ⭐ 1. LOCAL TIME TOOL
        # ─────────────────────────────
        if "time" in query_lower:
            now = datetime.datetime.now().strftime("%I:%M %p")
            Speaker().speak(f"The current time is {now}")
            return

        # ─────────────────────────────
        # ⭐ 2. LOCAL DATE TOOL
        # ─────────────────────────────
        if "date" in query_lower or ("today" in query_lower and "news" not in query_lower):
            today = datetime.datetime.now().strftime("%A, %d %B %Y")
            Speaker().speak(f"Today is {today}")
            return

        # ─────────────────────────────
        # ⭐ 3. WEATHER / TEMPERATURE TOOL
        # ─────────────────────────────
        if "temperature" in query_lower or "weather" in query_lower:
            try:
                # Example: Dombivli coordinates
                url = "https://api.open-meteo.com/v1/forecast?latitude=19.2183&longitude=73.0868&current_weather=true"
                data = requests.get(url, timeout=5).json()

                temp = data["current_weather"]["temperature"]
                Speaker().speak(f"The current temperature is {temp} degrees Celsius")
                return
            except Exception as e:
                logger.error(f"Weather API error: {e}")

        # ─────────────────────────────
        # ⭐ 4. WEB SEARCH FOR NEWS / INFO
        # ─────────────────────────────
        web_data = search_web(query)

        if not web_data or "Could not fetch" in web_data:
            Speaker().speak("Sorry, I could not fetch the latest information.")
            return

        # ─────────────────────────────
        # ⭐ 5. IMPROVED PROMPT
        # ─────────────────────────────
        prompt = f"""
You are a helpful voice assistant for visually impaired users.

Answer the user's question in MAXIMUM 2 short sentences.
Do NOT say phrases like "Here is a summary".
Speak naturally like a voice assistant.

Question: {query}
Information: {web_data}
"""

        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Clean speech formatting
        answer = answer.replace("\n", " ").strip()

        Speaker().speak(answer)

    except Exception as e:
        logger.error(f"Knowledge logic error: {e}", exc_info=True)
        Speaker().speak("Sorry, I couldn't process that.")