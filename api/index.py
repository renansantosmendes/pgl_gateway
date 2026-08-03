import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from tavily import TavilyClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("tavily-gateway")

app = FastAPI(title="Tavily Search Gateway")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

logger.info("Starting Tavily Search Gateway")
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY is not set in the environment")
else:
    logger.info("TAVILY_API_KEY loaded from environment")


@app.middleware("http")
async def log_requests(request: Request, call_next) -> object:
    """Log every incoming request and outgoing response, including timing.

    Args:
        request: The incoming HTTP request.
        call_next: The next handler in the middleware chain.

    Returns:
        The HTTP response produced by the downstream handler.
    """
    start_time = time.time()
    logger.info("Incoming request: %s %s", request.method, request.url.path)

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "Completed request: %s %s | status=%s | duration=%.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def get_client() -> TavilyClient:
    """Create and return a Tavily client using the API key from environment variables.

    Raises:
        HTTPException: if the TAVILY_API_KEY environment variable is not set.
    """
    logger.debug("Building Tavily client")

    if not TAVILY_API_KEY:
        logger.error("Cannot build Tavily client: TAVILY_API_KEY is missing")
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY is not configured.")

    logger.debug("Tavily client created successfully")
    return TavilyClient(api_key=TAVILY_API_KEY)


@app.get("/health")
def health() -> dict:
    """Return the service health status."""
    logger.info("Health check requested")
    status = {"status": "ok"}
    logger.info("Health check response: %s", status)
    return status


@app.get("/search")
def search(
    q: str = Query(..., description="Search query string"),
    search_depth: str = Query(
        "advanced", description="Tavily search depth: 'basic' or 'advanced'"
    ),
    include_raw_content: bool = Query(
        True, description="If true, include the full raw page content (not truncated)"
    ),
) -> dict:
    """Perform a search using the Tavily API and return the raw results.

    Args:
        q: The search query string.
        search_depth: Tavily search depth, 'basic' or 'advanced'.
        include_raw_content: Whether to include the untruncated raw page content.

    Returns:
        The full, unmodified response returned by the Tavily client.

    Raises:
        HTTPException: if the Tavily API key is missing or the request fails.
    """
    logger.info("Search requested with query: %r", q)

    client = get_client()

    start_time = time.time()
    try:
        logger.debug("Calling Tavily API")
        results = client.search(
            query=q,
            search_depth=search_depth,
            include_raw_content=include_raw_content,
        )
    except Exception as error:
        logger.exception("Tavily request failed for query: %r", q)
        raise HTTPException(status_code=502, detail=f"Tavily request failed: {error}")

    duration_ms = (time.time() - start_time) * 1000
    result_count = len(results.get("results", [])) if isinstance(results, dict) else 0
    logger.info(
        "Search completed for query: %r | results=%s | duration=%.2fms",
        q,
        result_count,
        duration_ms,
    )

    return results


@app.get("/debug/{full_path:path}", include_in_schema=False)
def debug_catch_all(full_path: str, request: Request) -> dict:
    """Temporary diagnostic route to inspect what path/scope Vercel forwards."""
    return {
        "debug": True,
        "full_path": full_path,
        "url_path": request.url.path,
        "url": str(request.url),
        "headers": dict(request.headers),
    }
