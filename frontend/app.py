import base64
from pathlib import Path
import sys
import io
from gtts import gTTS

# Ensure repository root is at the very front of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
while str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Remove frontend directory from sys.path if present, to prevent frontend/app.py
# from colliding with the top-level 'app' package.
_FRONTEND_DIR = str(PROJECT_ROOT / "frontend")
if _FRONTEND_DIR in sys.path:
    sys.path.remove(_FRONTEND_DIR)

# NOW import modules from app or rag
from app.services.visualization_service import analyze_and_plot
import httpx
import plotly.io as pio
import pandas as pd
from rag.document_loader import load_document
from rag.csv_processor import get_csv_summary, EmptyCSVError, CorruptedCSVError
import streamlit as st
from ui.results_card import render_results_card

st.set_page_config(page_title="Talking to Bridges", layout="wide")

API_BASE_URL = "http://127.0.0.1:8001"

st.title("Talking to Bridges")
st.subheader("An LLM-Based Intelligent Interface for Structural Health Monitoring")


# --- Excel Data Audit (Kolla) ---
try:
    from frontend.data_audit_ui import render_data_audit
    st.markdown("---")
    render_data_audit()
except Exception as e:
    st.error(f"Data Audit error: {e}")

# --- Clean Sensor Data Dashboard Page ---
try:
    from frontend.dashboard import render_dashboard
    st.markdown("---")
    render_dashboard(PROJECT_ROOT)
    st.markdown("---")
except Exception as e:  # noqa: BLE001
    st.error(f"Dashboard error: {e}")

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
        "Upload a PDF, DOCX, TXT, CSV, or sensor recording (.xlsx/.xlsm)",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xlsm"],
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

                    is_csv = uploaded_file.name.lower().endswith(".csv")
                    is_excel = uploaded_file.name.lower().endswith((".xlsx", ".xlsm"))

                    if is_excel:
                        # Sensor recording (.xlsx/.xlsm) — cleaned server-side
                        st.session_state.uploaded_doc = {
                            "filename": uploaded_file.name,
                            "document_id": data.get("document_id"),
                            "chunks_created": data.get("chunks_created"),
                            "file_type": Path(uploaded_file.name).suffix.lstrip("."),
                            "rows": data.get("rows"),
                            "columns": data.get("columns"),
                            "column_names": data.get("column_names", []),
                            "clean_data_path": data.get("clean_data_path"),
                            "preview_df": None,
                            "page_count": None,
                            "preview_text": None,
                        }
                    elif is_csv:
                        # For CSV: get a rich summary using the local csv_processor
                        try:
                            csv_summary = get_csv_summary(file_bytes, uploaded_file.name)
                        except (EmptyCSVError, CorruptedCSVError) as e:
                            csv_summary = None
                            st.warning(f"Could not generate local CSV preview: {e}")

                        st.session_state.uploaded_doc = {
                            "filename": uploaded_file.name,
                            "document_id": data.get("document_id"),
                            "chunks_created": data.get("chunks_created"),
                            "file_type": "csv",
                            "rows": data.get("rows"),
                            "columns": data.get("columns"),
                            "column_names": data.get("column_names", []),
                            "preview_df": csv_summary["preview_df"] if csv_summary else None,
                            "page_count": None,
                            "preview_text": None,
                        }
                    else:
                        # PDF / DOCX / TXT: existing local preview logic
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
                            "file_type": Path(uploaded_file.name).suffix.lstrip("."),
                            "page_count": page_count,
                            "preview_text": preview_text,
                            "rows": None,
                            "columns": None,
                            "column_names": [],
                            "preview_df": None,
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
        st.write(f"**File Type:** `{doc.get('file_type', 'unknown').upper()}`")
        st.write(f"**Chunks Created:** {doc['chunks_created']}")

        if doc.get("file_type") == "csv":
            # CSV-specific display
            if doc.get("rows") is not None:
                st.write(f"**Rows:** {doc['rows']}")
            if doc.get("columns") is not None:
                st.write(f"**Columns:** {doc['columns']}")
            if doc.get("column_names"):
                st.write(f"**Column Names:** {', '.join(doc['column_names'])}")

            if doc.get("preview_df") is not None:
                with st.expander("CSV Data Preview (first 10 rows)", expanded=True):
                    st.dataframe(doc["preview_df"], width='stretch')

        elif doc.get("file_type") in ("xlsx", "xlsm"):
            # Sensor recording — display cleaned-data summary
            if doc.get("rows") is not None:
                st.write(f"**Rows (cleaned):** {doc['rows']}")
            if doc.get("columns") is not None:
                st.write(f"**Columns:** {doc['columns']}")
            if doc.get("column_names"):
                st.write(f"**Column names:** {', '.join(doc['column_names'])}")
            if doc.get("clean_data_path"):
                st.write(f"**Clean data stored at:** `{doc['clean_data_path']}`")
            st.info(
                "✅ Sensor recording cleaned and indexed. "
                "You can now ask questions about this recording."
            )

        else:
            # PDF / DOCX / TXT display
            if doc.get("page_count") is not None:
                st.write(f"**Pages:** {doc['page_count']}")
            if doc.get("preview_text"):
                with st.expander("Extracted Text Preview", expanded=False):
                    st.markdown(doc["preview_text"])

# Main chat interface
for msg_idx, msg in enumerate(st.session_state.messages):
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

        if msg.get("assessment"):
            render_results_card(msg["assessment"], key=f"chat_card_{msg_idx}")
        if "fig" in msg:
            st.plotly_chart(msg["fig"], width='stretch', key=f"chat_fig_{msg_idx}")
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


if st.session_state.get("uploaded_doc") and st.session_state.uploaded_doc.get("preview_df") is not None:
    csv_df = st.session_state.uploaded_doc["preview_df"]
    numeric_cols = csv_df.select_dtypes(include=["number", "float64", "int64"]).columns.tolist()
    all_cols = csv_df.columns.tolist()

    if numeric_cols and len(all_cols) >= 2:
        with st.expander("Generate Interactive Chart & Voice Insights in Chat", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                time_col = st.selectbox("Select Time / X-Axis Column", options=all_cols, key="main_x_col")
            with col2:
                val_col = st.selectbox("Select Metric / Y-Axis Column", options=numeric_cols, key="main_y_col")

            if st.button("Generate Chart & Insights"):
                with st.spinner("Analyzing dataset and generating plot..."):
                    # 1. Generate plot figure and text explanation
                    fig, explanation = analyze_and_plot(csv_df, time_col, val_col)
                    
                    # 2. Compute key stats for speech output
                    max_val = csv_df[val_col].max()
                    min_val = csv_df[val_col].min()
                    avg_val = csv_df[val_col].mean()

                    spoken_text = (
                        f"Here is the chart for {val_col} over {time_col}. "
                        f"The maximum value is {max_val}, "
                        f"the minimum value is {min_val}, "
                        f"and the average value is {avg_val:.2f}."
                    )

                    # 3. Create audio bytes with gTTS
                    tts = gTTS(text=spoken_text, lang="en")
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    audio_bytes = fp.read()

                    # 4. Append directly to chat messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": explanation,
                        "fig": fig,
                        "audio": audio_bytes,
                        "content_type": "audio/mp3"
                    })

                    st.rerun()

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

                # Results card (NEW-5): same numbers as the answer, from the assessment field
                assessment = data.get("assessment")
                if assessment:
                    render_results_card(assessment, key=f"chat_card_new_{len(st.session_state.messages)}")

                # Render plot if backend generated one
                fig = None
                fig_json = data.get("fig")
                if fig_json:
                    fig = pio.from_json(fig_json)
                    st.plotly_chart(
                        fig,
                        width='stretch',
                        key=f"chat_fig_new_{len(st.session_state.messages)}",
                    )

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
                    speak_data = speak_resp.json()
                    audio_base64 = speak_data.get("audio_base64")
                    content_type = speak_data.get("content_type", "audio/wav")
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        st.audio(audio_bytes, format=content_type)

                # Store to history
                msg_data = {"role": "assistant", "content": answer, "sources": sources}
                if assessment:
                    msg_data["assessment"] = assessment
                if fig:
                    msg_data["fig"] = fig
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
