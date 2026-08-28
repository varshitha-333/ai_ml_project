"""
Main FastAPI Application Entrypoint with Lifespan Context Manager, CORS,
and Clean Exception Handling Guards.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.config import get_settings
from src.api.routes import router
from src.api.services import PipelineService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to preload static facet catalog & vector index ONCE on app startup.
    """
    print("FastAPI Lifespan Startup: Preloading pipeline singletons...")
    service = PipelineService.get_instance()
    service.initialize()
    yield
    print("FastAPI Lifespan Shutdown: Cleaning up resources.")


settings = get_settings()

app = FastAPI(
    title="Facet Evaluator API",
    description="Production-minded ML evaluation API for scoring conversational dialogue transcripts against behavioral facet catalogs.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Exception Handlers ensuring zero stack trace or secret exposure
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = errors[0]["msg"] if errors else "Invalid request payload format."
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Unprocessable Entity", "message": msg}
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Request Error", "message": str(exc.detail)}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Hide internal stack traces and secrets
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred during evaluation."}
    )


# Include API routes
app.include_router(router)


@app.get("/", include_in_schema=False)
def root_redirect():
    return {"message": "Facet Evaluator API active. Visit /docs for OpenAPI documentation."}
