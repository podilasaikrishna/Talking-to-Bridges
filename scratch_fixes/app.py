import base64
from pathlib import Path
import sys

# Ensure repository root is in python search path when running from Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from rag.document_loader import load_document
import streamlit as st

st.set_page_config(page_title="Talking to Bridges", layout="wide")

API_BASE_URL = "http://127.0.0.1:8001"

st.title("Talking to Bridges")
st.subheader("An LLM-Based Intelligent Interface for Structural Health Monitoring")

# Initialize session state for chat history and document details
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_doc" not in st.session_state:
    st.session_state.uploaded_doc = None
if "last_processed_audio_bytes" not in st.session_state:
    st.session_state.last_processed_audio_bytes = None

# Sidebar for document upload and status
with st.sidebar:
    st.header("Groq Connection")
    try:
        health_response = httpx.get(f"{API_BASE_URL}/health", timeout=120.0)
        if health_response.status_code == 200 and health_response.json().get(
            "groq_connected", False
        ):
            st.success("Groq connected")
        else:
            st.warning("Groq is not connected. Add GROQ_API_KEY to .env.")
    except httpx.HTTPError:
        st.warning("Backend is not running. Start the API before using Groq.")

    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"]
    )
    if st.button("Upload Document") and uploaded_file is not None:
        with st.spinner("Processing and indexing document..."):
            try:
                file_bytes = uploaded_file.getvalue()
                files = {
                    "file": (
                        uploaded_file.name,
                        file_bytes,
                        uploaded_file.type or "application/octet-stream",
                    )
                }
                response = httpx.post(f"{API_BASE_URL}/upload", files=files, timeout=120.0)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.messages = []

                    # Extract local preview & page count
                    try:
                        pages = load_document(uploaded_file.name, file_bytes)
                        page_count = len(pages)
                        preview_chunks = []
                        char_count = 0
                        for page in pages:
                            preview_chunks.append(f"**Page {page.page_number}**\n{page.text}")
                            char_count += len(page.text)
                            if char_count >= 3000:
                                break
                        preview_text = "\n\n---\n\n".join(preview_chunks)
                    except Exception:
                        page_count = 1
                        preview_text = file_bytes[:3000].decode("utf-8", errors="replace")

                    st.session_state.uploaded_doc = {
                        "filename": uploaded_file.name,
                        "document_id": data.get("document_id"),
                        "chunks_created": data.get("chunks_created"),
                        "page_count": page_count,
                        "preview_text": preview_text,
                    }
                    st.success("Document uploaded and indexed successfully!")
                else:
                    error_data = response.json()
                    err_msg = error_data.get("error", {}).get("message", response.text)
                    st.error(f"Upload failed: {err_msg}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Error connecting to server: {e}")

    # Display uploaded document details & preview
    if st.session_state.uploaded_doc:
        doc = st.session_state.uploaded_doc
        st.divider()
        st.subheader("Active Document Details")
        st.write(f"**File:** `{doc['filename']}`")
        st.write(f"**Document ID:** `{doc['document_id']}`")
        st.write(f"**Pages:** {doc['page_count']}")
        st.write(f"**Chunks Created:** {doc['chunks_created']}")

        with st.expander("Extracted Text Preview", expanded=False):
            st.markdown(doc["preview_text"])

# Main chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources & Citations", expanded=False):
                for idx, src in enumerate(msg["sources"], start=1):
                    source_file = src.get("source_file", "unknown")
                    page_num = src.get("page_number", 1)
                    st.markdown(f"**Source {idx}:** `{source_file}` (Page {page_num})")
                    if src.get("text"):
                        st.caption(src.get("text"))
        if "audio" in msg:
            st.audio(msg["audio"], format=msg.get("content_type", "audio/wav"))

user_query = None

text_prompt = st.chat_input("Ask a question about the bridge data...")
if text_prompt and text_prompt.strip():
    user_query = text_prompt.strip()
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

audio_value = st.audio_input("Or speak your question...")
if audio_value:
    raw_audio = audio_value.getvalue()
    if raw_audio and raw_audio != st.session_state.get("last_processed_audio_bytes"):
        st.session_state.last_processed_audio_bytes = raw_audio
        with st.spinner("Transcribing your voice..."):
            try:
                files = {"audio": ("mic_recording.wav", raw_audio, "audio/wav")}
                transcribe_resp = httpx.post(f"{API_BASE_URL}/transcribe", files=files, timeout=120.0)
                if transcribe_resp.status_code == 200:
                    transcribed_text = transcribe_resp.json().get("text", "").strip()
                    if transcribed_text:
                        user_query = transcribed_text
                        st.session_state.messages.append({"role": "user", "content": f"🎤 *{user_query}*"})
                        with st.chat_message("user"):
                            st.markdown(f"🎤 *{user_query}*")
                    else:
                        st.info("No speech detected in audio recording.")
                else:
                    err_msg = transcribe_resp.json().get("error", {}).get("message", "Failed to transcribe audio.")
                    st.error(f"STT Error: {err_msg}")
            except Exception as e:
                st.error(f"Error connecting to STT API: {e}")

if user_query:
    # Call backend for chat response
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            chat_resp = httpx.post(
                f"{API_BASE_URL}/api/chat",
                json={"message": user_query},
                timeout=120.0,
            )
            if chat_resp.status_code == 200:
                data = chat_resp.json()
                answer = data.get("answer", "No answer provided.")
                sources = data.get("sources", [])
                st.markdown(answer)

                if sources:
                    with st.expander("Sources & Citations", expanded=False):
                        for idx, src in enumerate(sources, start=1):
                            source_file = src.get("source_file", "unknown")
                            page_num = src.get("page_number", 1)
                            st.markdown(f"**Source {idx}:** `{source_file}` (Page {page_num})")
                            if src.get("text"):
                                st.caption(src.get("text"))

                # Call /speak for audio
                speak_resp = httpx.post(
                    f"{API_BASE_URL}/speak",
                    json={"message": answer},
                    timeout=120.0,
                )
                audio_bytes = None
                content_type = "audio/wav"
                if speak_resp.status_code == 200:
                    audio_base64 = speak_resp.json().get("audio_base64")
                    content_type = speak_resp.json().get("content_type", "audio/wav")
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        st.audio(audio_bytes, format=content_type)

                # Store to history
                msg_data = {"role": "assistant", "content": answer, "sources": sources}
                if audio_bytes:
                    msg_data["audio"] = audio_bytes
                    msg_data["content_type"] = content_type
                st.session_state.messages.append(msg_data)
            else:
                error_data = chat_resp.json()
                err_msg = error_data.get("error", {}).get("message", chat_resp.text)
                st.error(f"API Error: {err_msg}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to connect to backend API: {e}")

