import logging
import os
import time
from urllib.parse import parse_qsl, urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from tavily import TavilyClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("tavily-gateway")

fastapi_app = FastAPI(title="Tavily Search Gateway")


class VercelPathFixMiddleware:
    """Restore the real request path on Vercel's Python runtime.

    Vercel's Python ASGI runtime always sets scope["path"] to the serverless
    function's own file path (e.g. "/api/index") instead of the original
    request URL. vercel.json forwards the real path via the
    "original_path" query param so it can be restored here before routing.
    """

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive, send) -> None:
        if scope["type"] == "http":
            query_pairs = parse_qsl(scope.get("query_string", b"").decode())
            remaining_pairs = []
            for key, value in query_pairs:
                if key == "original_path":
                    scope["path"] = value or "/"
                    scope["raw_path"] = scope["path"].encode()
                else:
                    remaining_pairs.append((key, value))
            scope["query_string"] = urlencode(remaining_pairs).encode()
        await self.app(scope, receive, send)


app = VercelPathFixMiddleware(fastapi_app)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

logger.info("Starting Tavily Search Gateway")
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY is not set in the environment")
else:
    logger.info("TAVILY_API_KEY loaded from environment")


@fastapi_app.middleware("http")
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


@fastapi_app.get("/health")
def health() -> dict:
    """Return the service health status."""
    logger.info("Health check requested")
    status = {"status": "ok"}
    logger.info("Health check response: %s", status)
    return status


@fastapi_app.get("/search")
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
