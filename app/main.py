import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import analyze

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Link Analysis API",
    description="Analyzes a website with an LLM: competitors, search queries, SEO, services and audience.",
    version="1.0.0",
)


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad URL or website problem"},
        422: {"model": ErrorResponse, "description": "Invalid request body"},
        502: {"model": ErrorResponse, "description": "LLM failure"},
    },
)
def link_analysis(request: LinkAnalysisRequest):
    return analyze(request.url)


# ---------------------------------------------------------------- error handlers
@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError):
    # raised by the validator and the extractor (bad URL, unreachable site...)
    return error_response(400, "INVALID_REQUEST", str(exc))


@app.exception_handler(StarletteHTTPException)
def http_error_handler(request: Request, exc: StarletteHTTPException):
    # e.g. the 502 raised in analysis_service, and FastAPI's own 404/405
    return error_response(exc.status_code, f"HTTP_{exc.status_code}", str(exc.detail))


@app.exception_handler(RequestValidationError)
def validation_error_handler(request: Request, exc: RequestValidationError):
    # body missing or "url" not a string
    return error_response(422, "INVALID_BODY", 'Send a JSON body like {"url": "https://example.com"}.')


@app.exception_handler(Exception)
def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error")
    return error_response(500, "INTERNAL_ERROR", "Something went wrong on our side.")