import os
import re

path = 'app/api/routes.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()

# uuid
content = content.replace('import wave', 'import uuid\nimport wave')
content = content.replace('    document_id = f"doc-{len(_uploaded_documents) + 1:04d}"', '    document_id = f"doc-{uuid.uuid4().hex[:8]}"')

# ChatRequest history
content = content.replace('class ChatRequest(BaseModel):\n    message: str = Field(\n        ..., min_length=1, description="User query or prompt message"\n    )', 'class Message(BaseModel):\n    role: str\n    content: str\n\nclass ChatRequest(BaseModel):\n    message: str = Field(\n        ..., min_length=1, description="User query or prompt message"\n    )\n    chat_history: list[Message] = Field(default_factory=list)')

# upload
content = re.sub(r'        DOCUMENTS_DIR\.mkdir.*?_uploaded_documents\.clear\(\)', '''        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        import os
        safe_filename = os.path.basename(file.filename)
        file_path = DOCUMENTS_DIR / f"{document_id}_{safe_filename}"
        
        temp_vector_store = FAISSVectorStore()
        ingest_result = await asyncio.to_thread(
            ingest_document,
            file.filename,
            content,
            document_id,
            embedding_service,
            temp_vector_store,
        )
        global vector_store
        vector_store = temp_vector_store
        retriever.vector_store = vector_store

    except (UnsupportedFileTypeError, EmptyDocumentError, DocumentExtractionError, DocumentHandlingError, ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except (EmbeddingError, VectorStoreError) as error:
        raise HTTPException(status_code=500, detail=str(error))

    _uploaded_documents.clear()''', content, flags=re.DOTALL)

# transcribe
content = re.sub(r'        # Fallback to dummy if something breaks\n        return TranscribeResponse\(text="This is dummy transcribed speech \(fallback due to error\)\."\)', '        raise HTTPException(status_code=500, detail="Internal error")', content)
content = content.replace('transcribed_text = run_stt(temp_path)', 'transcribed_text = await asyncio.to_thread(run_stt, temp_path)')

# speak
content = re.sub(r'        # Fallback to silent wav if everything breaks\n        return SpeakResponse\(audio_base64=_dummy_wav_base64\(\), content_type="audio/wav"\)', '        raise HTTPException(status_code=500, detail="Internal error")', content)

# chat
content = re.sub(r'        if retrieved_chunks:.*?(?=        return ChatResponse\()', '''        if retrieved_chunks:
            context = "\\n\\n".join(
                f"[Source: {result.chunk.source_file}, page {result.chunk.page_number}]\\n{result.chunk.text}"
                for result in retrieved_chunks
            )
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
''', content, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f: f.write(content)
print("Patching complete part 2!")
