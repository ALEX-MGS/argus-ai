from openai import AsyncOpenAI
from app.core.config import settings


class EmbeddingService:

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.API_KEY)
        self.model = "text-embedding-3-small"

    async def embed(self, text: str) -> list:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
