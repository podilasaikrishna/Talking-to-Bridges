import os
import re

path = 'app/api/routes.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()

# 1. Update imports
content = content.replace('import wave', 'import uuid\nimport wave')

# 2. Update ChatRequest
content = content.replace('class ChatRequest(BaseModel):\n    message: str = Field(\n        ..., min_length=1, description="User query or prompt message"\n    )', 'class Message(BaseModel):\n    role: str\n    content: str\n\nclass ChatRequest(BaseModel):\n    message: str = Field(\n        ..., min_length=1, description="User query or prompt message"\n    )\n    chat_history: list[Message] = Field(default_factory=list)')

# 3. Replace upload
upload_pattern = r'@router\.post\("/upload", response_model=UploadResponse\).*?(?=\ndef _get_all_chunks)'
new_upload = '''@router.post("/upload", response_model=UploadResponse)
async def upload(file: Annotated[UploadFile, File(...)]):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A file name is required",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".csv"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Supported formats are: .pdf, .docx, .txt, .csv",
        )

    document_id = f"doc-{uuid.uuid4().hex[:8]}"
    try:
        content = await file.read()
        
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
            
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        import os
        safe_filename = os.path.basename(file.filename)
        file_path = DOCUMENTS_DIR / f"{document_id}_{safe_filename}"
        
        temp_vector_store = FAISSVectorStore()
        
        if suffix == ".csv":
            import asyncio
            ingest_result = await asyncio.to_thread(
                ingest_csv,
                file.filename,
                content,
                document_id,
                embedding_service,
                temp_vector_store,
            )
        else:
            import asyncio
            ingest_result = await asyncio.to_thread(
                ingest_document,
                file.filename,
                content,
                document_id,
                embedding_service,
                temp_vector_store,
            )
            
        global vector_store
        vector_store.index = temp_vector_store.index
        vector_store.dimension = temp_vector_store.dimension
        vector_store.metadata = temp_vector_store.metadata

    except (UnsupportedFileTypeError, EmptyDocumentError, DocumentExtractionError, DocumentHandlingError, ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except (EmptyCSVError, CorruptedCSVError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except (EmbeddingError, VectorStoreError) as error:
        raise HTTPException(status_code=500, detail=str(error))

    _uploaded_documents.clear()
    chunks = [
        ChunkResponse(
            text=chunk.text,
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
        )
        for chunk in ingest_result.chunks
    ]
    _uploaded_documents[document_id] = chunks

    return UploadResponse(
        document_id=document_id,
        chunks_created=ingest_result.chunks_created,
        file_type=suffix.lstrip("."),
        rows=ingest_result.rows if hasattr(ingest_result, 'rows') else None,
        columns=ingest_result.columns if hasattr(ingest_result, 'columns') else None,
        column_names=ingest_result.column_names if hasattr(ingest_result, 'column_names') else None,
    )
'''
content = re.sub(upload_pattern, new_upload, content, flags=re.DOTALL)

# 4. Replace transcribe_audio
content = content.replace('transcribed_text = run_stt(temp_path)', 'import asyncio\n            transcribed_text = await asyncio.to_thread(run_stt, temp_path)')
content = re.sub(r'        # Fallback to dummy if something breaks\n        return TranscribeResponse\(text="This is dummy transcribed speech \(fallback due to error\)\."\)', '        raise HTTPException(status_code=500, detail="Internal error")', content)

# 5. Replace chat
chat_pattern = r'@router\.post\("/chat", response_model=ChatResponse\).*'
new_chat = '''@router.post("/chat", response_model=ChatResponse)
@router.post("/api/chat", response_model=ChatResponse, include_in_schema=False)
async def chat(request: ChatRequest):
    clean_message = request.message.strip()
    if not clean_message:
        raise HTTPException(status_code=400, detail="Message empty")

    try:
        retrieved_chunks = []
        try:
            semantic_results = retriever.retrieve(clean_message, top_k=3)
            if semantic_results:
                retrieved_chunks = [RetrievalResult(chunk=ChunkResponse(text=r.text, source_file=r.source_file, page_number=r.page_number, chunk_id=r.chunk_id), score=r.score) for r in semantic_results]
        except Exception:
            pass
            
        if not retrieved_chunks:
            retrieved_chunks = _lexical_document_results(clean_message, limit=3)

        if retrieved_chunks:
            context = "\\n\\n".join(f"[Source: {r.chunk.source_file}]\\n{r.chunk.text}" for r in retrieved_chunks)
            from rag.prompts import SYSTEM_PROMPT, USER_PROMPT
            system_content = SYSTEM_PROMPT.format(context=context)
            user_content = USER_PROMPT.format(query=clean_message)
        else:
            system_content = "You are a helpful assistant for the Talking to Bridges platform. The user has not provided any document context. If they ask about a document, inform them that none is uploaded."
            user_content = clean_message

        messages = [{"role": "system", "content": system_content}]
        for msg in request.chat_history[-5:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_content})

        answer = await llm_service.generate(messages)
        return ChatResponse(answer=answer, sources=[r.chunk for r in retrieved_chunks])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
content = re.sub(chat_pattern, new_chat, content, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f: f.write(content)
print("Patching routes.py complete!")
