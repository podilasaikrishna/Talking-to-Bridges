import os
import re

# 1. document_loader.py
path = 'rag/document_loader.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()
content = re.sub(r'    except Exception:\n\s+# Fallback for plain-text mock data passed in unit tests\n\s+try:.*?(?=\n\s+if not pages:)', r'    except Exception as e:\n        raise DocumentExtractionError(f"Failed to read PDF file \'{filename}\': {e}") from e\n', content, flags=re.DOTALL)
content = re.sub(r'    except Exception:\n\s+# Fallback for plain-text mock data\n\s+try:.*?(?=\n\s+combined_text = )', r'    except Exception as e:\n        raise DocumentExtractionError(f"Failed to read DOCX file \'{filename}\': {e}") from e\n', content, flags=re.DOTALL)
with open(path, 'w', encoding='utf-8') as f: f.write(content)

# 2. vector_store.py
path = 'rag/vector_store.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()
content = content.replace('self.index = None\n        self.metadata = []', 'self.index = None\n        self.dimension = None\n        self.metadata = []')
with open(path, 'w', encoding='utf-8') as f: f.write(content)

# 3. config.py
path = 'app/core/config.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()
content = content.replace('CHUNK_SIZE: int = 1000', 'CHUNK_SIZE: int = 500').replace('CHUNK_OVERLAP: int = 150', 'CHUNK_OVERLAP: int = 50')
with open(path, 'w', encoding='utf-8') as f: f.write(content)

# 4. llm_service.py
path = 'app/services/llm_service.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()
content = content.replace('GrokServiceError', 'GroqServiceError')
content = re.sub(r'    async def generate\(self, prompt: str\) -> str:\n.*?return choices\[0\]\["message"\]\["content"\]\n        except.*?Failed to communicate with Groq"\)', '''    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from the configured Groq model."""
        import asyncio
        if not self.api_key:
            raise GroqUnavailableError("Groq API key is not configured")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    
                    if response.status_code == 429:
                        if attempt < max_retries - 1:
                            backoff = 2 ** attempt
                            logger.warning(f"Groq rate limit hit. Retrying in {backoff}s...")
                            await asyncio.sleep(backoff)
                            continue
                        else:
                            raise GroqServiceError("Groq rate limit exceeded (429)")

                    if response.status_code in (401, 403):
                        raise GroqUnavailableError("Groq API key is invalid")
                    if response.status_code == 404:
                        raise GroqModelNotFoundError(f"Model '{self.model}' was not found.")
                    if response.status_code != 200:
                        raise GroqServiceError(f"Groq API error status: {response.status_code}")

                    choices = response.json().get("choices", [])
                    if not choices or not choices[0].get("message", {}).get("content"):
                        raise GroqServiceError("Groq returned an empty response")
                    return choices[0]["message"]["content"]
                    
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RequestError) as error:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if isinstance(error, httpx.TimeoutException):
                    raise GroqServiceError("Groq request timed out")
                raise GroqUnavailableError("Groq API is unavailable")''', content, flags=re.DOTALL)
with open(path, 'w', encoding='utf-8') as f: f.write(content)

# 5. stt_service.py
path = 'app/services/stt_service.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()
content = re.sub(r'# GLOBAL MODEL INITIALIZATION.*?(?=def transcribe\()', '''_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading Speech-to-Text Model (faster-whisper, base, GPU)...")
        try:
            _model = WhisperModel("base", device="cuda", compute_type="float16")
        except Exception as e:
            print(f"GPU load failed ({e}). Falling back to CPU...")
            _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model

''', content, flags=re.DOTALL)
content = content.replace('    if not os.path.exists(audio_file_path):\n        return "Error: Audio file not found."\n        \n    # Transcribe the audio\n    segments, info = model.transcribe(audio_file_path, beam_size=5)', '    if not os.path.exists(audio_file_path):\n        raise FileNotFoundError("Error: Audio file not found.")\n        \n    model_instance = get_model()\n    segments, info = model_instance.transcribe(audio_file_path, beam_size=5, vad_filter=True)')
with open(path, 'w', encoding='utf-8') as f: f.write(content)

# 6. tts_service.py
path = 'app/services/tts_service.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()
content = content.replace('requests.post(url, json=payload, headers=headers)', 'requests.post(url, json=payload, headers=headers, timeout=10.0)')
content = re.sub(r'import urllib\.request.*?(?=_piper_voice = PiperVoice\.load)', '''import requests
            import tempfile
            import shutil
            
            model_name = "en_US-lessac-high"
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/{model_name}"
            model_path = os.path.join(os.path.dirname(__file__), f"{model_name}.onnx")
            config_path = f"{model_path}.json"
            
            def download_file(url, out_path):
                print(f"Downloading {out_path}...")
                with requests.get(url, stream=True, timeout=30.0) as r:
                    r.raise_for_status()
                    fd, tmp = tempfile.mkstemp()
                    os.close(fd)
                    with open(tmp, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    shutil.move(tmp, out_path)
            
            if not os.path.exists(model_path):
                download_file(f"{base_url}.onnx", model_path)
                download_file(f"{base_url}.onnx.json", config_path)
            
            ''', content, flags=re.DOTALL)
with open(path, 'w', encoding='utf-8') as f: f.write(content)

# 7. main.py
path = 'app/main.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()
content = content.replace('allow_origins=["*"]', 'allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:8000", "http://127.0.0.1:8000"]')
content = content.replace('app.include_router(api_router)', 'from fastapi.staticfiles import StaticFiles\napp.mount("/ui", StaticFiles(directory="frontend"), name="ui")\napp.include_router(api_router)')
with open(path, 'w', encoding='utf-8') as f: f.write(content)

# 8. prompts.py
path = 'rag/prompts.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()
content = content.replace('Context:\n{context}\n"""\n\nUSER_PROMPT = """Question: {query}"""', '<context>\n{context}\n</context>\n"""\n\nUSER_PROMPT = """{query}"""')
with open(path, 'w', encoding='utf-8') as f: f.write(content)

# 9. retrieval.py
path = 'rag/retrieval.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()
content = content.replace('        for chunk_dict, score in raw_results:\n            results.append(', '        for chunk_dict, score in raw_results:\n            if score < 0.35: continue\n            results.append(')
with open(path, 'w', encoding='utf-8') as f: f.write(content)

print("Patching complete part 1!")
