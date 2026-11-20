import logging
import threading

from cachetools import TTLCache
from fastapi import HTTPException

from app.schemas import LinkAnalysisResponse
from app.services.llm_service import analyze_content
from app.services.website_extractor import extract_website
from app.utils.validator import validate_url
from app.config import settings

logger = logging.getLogger(__name__)

_cache = TTLCache(maxsize=settings.CACHE_MAX_ITEMS, ttl=settings.CACHE_TTL_SECONDS)
_cache_lock = threading.Lock()

COMPETITOR_NOTE = "Competitors are AI-generated suggestions"

def analyze(raw_url: str)-> LinkAnalysisResponse:
    # 1. validate and normalize the URL 
    url = validate_url(raw_url)


    # 2. cache lookup
    with _cache_lock:
        cached = _cache.get(url)
    if cached is not None:
        logger.info(f"[analysis] cache hit | {url}")
        return cached

    # 3. fetch and clean the website (raises ValueError on website problems)
    data = extract_website(url)

    # 4. LLM call, with its own error handling
    try:
        result = analyze_content(data)
    except RuntimeError as e:
        logger.error(f"[analysis] LLM call failed | {e}")
        raise HTTPException(status_code=502, detail="The AI service failed. Please try again.")
    except ValueError as e:
        logger.error(f"[analysis] LLM output invalid | {e}")
        raise HTTPException(status_code=502, detail="The AI returned an invalid response. Please try again.")

    # 5. always label competitors as AI suggestions
    result.analysis.competitor_analysis.note = COMPETITOR_NOTE

    # 6. build the response
    response = LinkAnalysisResponse(
        website=data["url"],
        brand=result.brand,
        analysis=result.analysis,
    )

    # 7. store (only successes reach this line) and return
    with _cache_lock:
        _cache[url] = response
    return response