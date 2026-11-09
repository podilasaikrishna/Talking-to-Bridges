from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router as api_router, vector_store
from app.core.config import settings
from app.core.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Talking to Bridges API Server")
    logger.info(f"Groq Target URL: {settings.GROQ_BASE_URL}")
    logger.info(f"Groq Target Model: {settings.GROQ_MODEL}")
    # Load persistent vector database if available
    loaded = vector_store.load()
    if loaded:
        logger.info(f"Vector store loaded with {vector_store.total_vectors} chunks")
    else:
        logger.info("Vector store initialized (empty)")
    yield
    logger.info("Shutting down Talking to Bridges API Server")


app = FastAPI(
    title="Talking to Bridges API",
    description="An LLM-Based Intelligent Interface for Structural Health Monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", "http://127.0.0.1:8501",
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:5500", "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

_VOICE_UI = Path(__file__).resolve().parent.parent / "frontend" / "voice_ui.html"


@app.get("/ui", include_in_schema=False)
@app.get("/ui/voice_ui.html", include_in_schema=False)
async def voice_ui():
    """Serve the hands-free voice page from the same origin as the API."""
    if not _VOICE_UI.exists():
        return JSONResponse(status_code=404, content={"error": {"message": "voice_ui.html not found", "type": "NotFound"}})
    return FileResponse(_VOICE_UI)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": str(exc.detail), "type": "HTTPException"}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": {"message": "Invalid request parameters", "details": exc.errors(), "type": "ValidationError"}},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled application error")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "InternalError"}},
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
    )
