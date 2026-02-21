# modules/knowledge/knowledge_tool.py

from duckduckgo_search import DDGS
import time
from utils.logger import logger


def search_web(query: str, max_results: int = 3):
    try:
        time.sleep(1)  # ⭐ avoid rate limit

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, backend="lite"))

        if not results:
            return "No information found."

        snippets = []
        for r in results:
            snippets.append(r.get("body", ""))

        return " ".join(snippets)

    except Exception as e:
        logger.error(f"Knowledge search error: {e}")
        return "Could not fetch latest information."