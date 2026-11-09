import os
import re

path = 'app/services/llm_service.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()

content = re.sub(r'    async def generate\(self, prompt: str\) -> str:\n.*?return choices\[0\]\["message"\]\["content"\]\n        except.*?Failed to communicate with Groq"\)', '''    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from the configured Groq model."""
        import asyncio
        import httpx
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
print("Patching llm_service.py complete!")
