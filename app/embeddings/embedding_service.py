from __future__ import annotations

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

    async def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list]:
        """Genera embeddings para varios textos por lote.

        La API acepta un array de entradas, así que indexar un corpus de N
        fragmentos cuesta N/batch_size llamadas en vez de N. El orden de salida
        corresponde al de entrada.
        """
        if not texts:
            return []

        vectors: list[list] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            response = await self.client.embeddings.create(
                model=self.model,
                input=batch
            )

            # La API no garantiza el orden de `data`; se ordena por índice.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)

        return vectors
