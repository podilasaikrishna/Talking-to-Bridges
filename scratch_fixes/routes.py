import asyncio
import base64
import io
import os
import re
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.logging_config import logger
from app.services.llm_service import (
    GroqLLMService,
    GroqModelNotFoundError,
    GroqServiceError,
    GroqUnavailableError,
)
from rag.document_loader import (
    DocumentExtractionError,
    DocumentHandlingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from rag.embeddings import EmbeddingError, EmbeddingService
from rag.ingestion import DOCUMENTS_DIR, ingest_document
from rag.prompts import SYSTEM_PROMPT
from rag.retrieval import VectorRetriever
from rag.vector_store import FAISSVectorStore, VectorStoreError

router = APIRouter()
llm_service = GroqLLMService()
embedding_service = EmbeddingService()
vector_store = FAISSVectorStore()
retriever = VectorRetriever(embedding_service=embedding_service, vector_store=vector_store)

_uploaded_documents: dict[str, list["ChunkResponse"]] = {}
_upload_lock = asyncio.Lock()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_HISTORY_MESSAGES = 6
OVERVIEW_CHUNK_LIMIT = 8
OVERVIEW_HINTS = (
    "summar", "overview", "describe", "main points", "key points",
    "what is this", "what's this", "about this", "tell me about",
)

NO_MATCH_PROMPT = (
    "You are a helpful, precise assistant for the Talking-to-Bridges platform.\n"
    "A bridge document is loaded, but no passage in it matched the user's message.\n"
    "If the message is a greeting or small talk, reply politely and invite questions "
    "about the bridge document. Otherwise, say that the uploaded document does not "
    "appear to contain that information. Do not invent facts about the bridge."
)
NO_DOCUMENT_PROMPT = (
    "You are a helpful assistant for the Talking-to-Bridges platform. "
    "No document has been uploaded yet. If the user greets you, reply politely. "
    "If they ask about a bridge document, tell them to upload one first."
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User query or prompt message")
    chat_history: list[Message] = Field(default_factory=list)


class ChunkResponse(BaseModel):
    text: str
    source_file: str
    page_number: int
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChunkResponse] = Field(default_factory=list)


class UploadResponse(BaseModel):
    document_id: str
    chunks_created: int


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    num_results: int = Field(default=3, ge=1, le=20)


class RetrievalResult(BaseModel):
    chunk: ChunkResponse
    score: float


class RetrieveResponse(BaseModel):
    results: list[RetrievalResult]


class TranscribeResponse(BaseModel):
    text: str


class SpeakResponse(BaseModel):
    audio_base64: str
    content_type: str = "audio/wav"


class HealthResponse(BaseModel):
    status: str = "ok"
    groq_connected: bool = False


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint to verify backend status and Groq availability."""
    groq_status = await llm_service.health_check()
    return HealthResponse(status="ok", groq_connected=groq_status)


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------
@router.post("/upload", response_model=UploadResponse)
async def upload(file: Annotated[UploadFile, File(...)]):
    """Extract, clean, chunk, embed, and store a document in the FAISS vector database.

    The new document is built in a staging store and only replaces the active one
    after ingestion fully succeeds, so a failed upload never wipes the current document.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A file name is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Supported formats are: .pdf, .docx, .txt",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )

    document_id = f"doc-{uuid.uuid4().hex[:8]}"

    async with _upload_lock:
        try:
            staging_store = FAISSVectorStore()
            ingest_result = await asyncio.to_thread(
                ingest_document,
                file.filename,
                content,
                document_id,
                embedding_service,
                staging_store,
            )
        except (
            UnsupportedFileTypeError,
            EmptyDocumentError,
            DocumentExtractionError,
            DocumentHandlingError,
            ValueError,
            OSError,
        ) as error:
            logger.error(f"Document upload error for '{file.filename}': {error}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
        except (EmbeddingError, VectorStoreError) as error:
            logger.error(f"RAG processing error for '{file.filename}': {error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process and index document: {error}",
            ) from error

        # Success: swap the staged index into the live store (no awaits in between).
        vector_store.index = staging_store.index
        vector_store.dimension = staging_store.dimension
        vector_store.metadata = staging_store.metadata

        _uploaded_documents.clear()
        _uploaded_documents[document_id] = [
            ChunkResponse(
                text=chunk.text,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
            )
            for chunk in ingest_result.chunks
        ]

    # Keep a copy of the source file (best effort - never fails the upload).
    try:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename.replace("\\", "/")).name
        (DOCUMENTS_DIR / f"{document_id}_{safe_name}").write_bytes(content)
    except OSError as error:
        logger.warning(f"Could not save a copy of '{file.filename}': {error}")

    return UploadResponse(document_id=document_id, chunks_created=ingest_result.chunks_created)


# --------------------------------------------------------------------------
# Retrieval helpers
# --------------------------------------------------------------------------
def _get_all_chunks() -> list[ChunkResponse]:
    """All chunks of the active document (in-memory state, else persisted metadata)."""
    if _uploaded_documents:
        return [chunk for document in _uploaded_documents.values() for chunk in document]
    if vector_store.metadata:
        return [
            ChunkResponse(
                text=m.get("text", ""),
                source_file=m.get("source_file", ""),
                page_number=m.get("page_number", 1),
                chunk_id=m.get("chunk_id", ""),
            )
            for m in vector_store.metadata
        ]
    return []


def _lexical_document_results(question: str, limit: int = 3) -> list[RetrievalResult]:
    """Keyword fallback, used only when semantic search itself fails."""
    chunks = _get_all_chunks()
    terms = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
    if not chunks or not terms:
        return []

    scored = []
    for chunk in chunks:
        chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
        matched = terms & chunk_terms
        if matched:
            scored.append(RetrievalResult(chunk=chunk, score=len(matched) / len(terms)))
    scored.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
    return scored[:limit]


def _semantic_results(question: str, top_k: int, min_score: float | None = None) -> list[RetrievalResult]:
    """Blocking semantic search - always run this in a worker thread."""
    return [
        RetrievalResult(
            chunk=ChunkResponse(
                text=r.text,
                source_file=r.source_file,
                page_number=r.page_number,
                chunk_id=r.chunk_id,
            ),
            score=r.score,
        )
        for r in retriever.retrieve(question, top_k=top_k, min_score=min_score)
    ]


def _overview_results() -> list[RetrievalResult]:
    """First chunks of the document, in order - used for 'summarize this' style questions."""
    return [
        RetrievalResult(chunk=chunk, score=1.0)
        for chunk in _get_all_chunks()[:OVERVIEW_CHUNK_LIMIT]
    ]


def _wants_overview(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered for hint in OVERVIEW_HINTS)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant document chunks using FAISS semantic vector search."""
    clean_question = request.question.strip()
    if not clean_question:
        return RetrieveResponse(results=[])

    try:
        results = await asyncio.to_thread(_semantic_results, clean_question, request.num_results, 0.0)
        return RetrieveResponse(results=results)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Vector search failed, falling back to lexical search: {e}")
        return RetrieveResponse(results=_lexical_document_results(clean_question, request.num_results))


# --------------------------------------------------------------------------
# Speech
# --------------------------------------------------------------------------
@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: Annotated[UploadFile, File(...)]):
    """Transcribe audio with faster-whisper. Failures are real errors, never fake text."""
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty.")

    try:
        from app.services.stt_service import transcribe as run_stt
    except ImportError as error:
        logger.error(f"Speech-to-text engine unavailable: {error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech-to-text is unavailable: faster-whisper is not installed.",
        ) from error

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(temp_fd, "wb") as temp_file:
            temp_file.write(content)
        text = await asyncio.to_thread(run_stt, temp_path)
    except Exception as error:  # noqa: BLE001
        logger.exception("Transcription failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {error}",
        ) from error
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return TranscribeResponse(text=text)


def _silent_wav_base64() -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@router.post("/speak", response_model=SpeakResponse)
async def speak(request: ChatRequest):
    """Convert text to speech using the active TTS engine (ElevenLabs, Edge, or Piper)."""
    clean_message = request.message.strip()
    if not clean_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")
    try:
        from app.services.tts_service import generate_speech

        audio_b64, content_type = await asyncio.to_thread(generate_speech, clean_message)
        return SpeakResponse(audio_base64=audio_b64, content_type=content_type)
    except Exception as error:  # noqa: BLE001
        # Kept deliberately: voice_ui.html expects a playable payload, so a TTS failure
        # degrades to silence (the text answer is still shown). The error is logged loudly.
        logger.error(f"TTS failed, returning silent audio: {error}")
        return SpeakResponse(audio_base64=_silent_wav_base64(), content_type="audio/wav")


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse)
@router.post("/api/chat", response_model=ChatResponse, include_in_schema=False)
async def chat(request: ChatRequest):
    """Retrieve relevant chunks and generate a grounded answer from Groq."""
    clean_message = request.message.strip()
    if not clean_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty or whitespace only",
        )

    try:
        results: list[RetrievalResult] = []
        try:
            if _wants_overview(clean_message):
                results = _overview_results()
            else:
                results = await asyncio.to_thread(_semantic_results, clean_message, 3)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Vector retrieval failed, using keyword fallback: {e}")
            results = _lexical_document_results(clean_message, limit=3)

        has_document = vector_store.total_vectors > 0 or bool(_uploaded_documents)

        if results:
            context = "\n\n".join(
                f"[Source: {r.chunk.source_file}, page {r.chunk.page_number}]\n{r.chunk.text}"
                for r in results
            )
            system_content = SYSTEM_PROMPT.format(context=context)
        elif has_document:
            system_content = NO_MATCH_PROMPT
        else:
            system_content = NO_DOCUMENT_PROMPT

        messages = [{"role": "system", "content": system_content}]
        for item in request.chat_history[-MAX_HISTORY_MESSAGES:]:
            messages.append({"role": item.role, "content": item.content})
        messages.append({"role": "user", "content": clean_message})

        answer = await llm_service.generate(messages)
        return ChatResponse(answer=answer, sources=[r.chunk for r in results])

    except GroqUnavailableError:
        logger.error("Chat endpoint error: Groq service is unavailable or the API key is invalid")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service is unavailable (check GROQ_API_KEY and your internet connection)",
        )
    except GroqModelNotFoundError as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except GroqServiceError as e:
        logger.error(f"Chat endpoint error: {e}")
        if "429" in str(e):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq rate limit reached. Please wait a few seconds and try again.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal service error communicating with LLM",
        )
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error in chat endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
