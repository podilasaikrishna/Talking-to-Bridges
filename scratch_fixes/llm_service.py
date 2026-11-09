import httpx

from app.core.config import settings
from app.core.logging_config import logger


class GroqServiceError(Exception):
    """Base exception for Groq API failures."""


class GroqUnavailableError(GroqServiceError):
    """Raised when the Groq API is unreachable or credentials are missing."""


class GroqModelNotFoundError(GroqServiceError):
    """Raised when the configured Groq model is not available."""


class GroqLLMService:
    """Service abstraction for Groq's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or settings.GROQ_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def health_check(self) -> bool:
        """Check whether the configured Groq API credentials and endpoint work."""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/models", headers=self.headers
                )
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as error:
            logger.warning(f"Groq health check failed: {error}")
            return False

    async def list_models(self) -> list[str]:
        """Return model IDs available to the configured Groq API key."""
        if not self.api_key:
            raise GroqUnavailableError("Groq API key is not configured")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/models", headers=self.headers
                )
                if response.status_code == 401:
                    raise GroqUnavailableError("Groq API key is invalid")
                if response.status_code != 200:
                    raise GroqServiceError(
                        f"Groq model listing failed with status {response.status_code}"
                    )
                return [
                    model["id"]
                    for model in response.json().get("data", [])
                    if model.get("id")
                ]
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            logger.error(f"Connection failure to Groq at {self.base_url}: {error}")
            raise GroqUnavailableError("Groq API is unavailable")
        except httpx.TimeoutException as error:
            logger.error(f"Groq model listing timed out: {error}")
            raise GroqServiceError("Groq model listing timed out")
        except httpx.RequestError as error:
            logger.error(f"Groq model listing request failed: {error}")
            raise GroqServiceError("Failed to list Groq models")

    async def generate(self, messages: "str | list[dict[str, str]]") -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
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
                raise GroqUnavailableError("Groq API is unavailable")
