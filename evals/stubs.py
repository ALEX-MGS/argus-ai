"""Dobles deterministas de los servicios que cuestan dinero.

Sirven para verificar que la tubería completa —carga, fragmentación, indexado,
recuperación, prompt, puntuación— funciona de punta a punta sin llamar a ninguna
API. Permiten depurar el harness gratis y que la primera corrida real no se vaya
en errores de plomería.

**No miden calidad.** Los embeddings son bolsa de palabras con hashing, así que
la recuperación es léxica, no semántica; y el modelo no razona, solo devuelve el
contexto que recibió. Los números de una corrida con dobles solo dicen que las
piezas encajan.
"""

from __future__ import annotations

import json
import math
import re
from zlib import crc32

from app.models.base_llm import BaseLLM


DEFAULT_DIMENSION = 1536

_TOKEN = re.compile(r"\w+", re.UNICODE)


def hashed_embedding(text: str, dimension: int = DEFAULT_DIMENSION) -> list[float]:
    """Vector de bolsa de palabras con hashing, normalizado a norma 1.

    Se usa `crc32` y no `hash()` porque el hash de strings de Python está
    aleatorizado por proceso: con él, dos corridas darían índices distintos.

    Al quedar normalizado, la distancia L2 al cuadrado cae en [0, 4], igual que
    con los embeddings de OpenAI. Eso hace que el umbral del pipeline se comporte
    de forma comparable a la real.
    """
    vector = [0.0] * dimension

    for token in _TOKEN.findall(text.lower()):
        vector[crc32(token.encode("utf-8")) % dimension] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))

    if norm == 0.0:
        # Sin tokens no hay dirección posible; se devuelve un vector unitario
        # fijo para no romper la normalización aguas abajo.
        vector[0] = 1.0
        return vector

    return [value / norm for value in vector]


class StubEmbeddingService:
    """Reemplazo de `EmbeddingService` que no llama a la API."""

    def __init__(self, dimension: int = DEFAULT_DIMENSION):
        self.dimension = dimension
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return hashed_embedding(text, self.dimension)

    async def embed_batch(
        self, texts: list[str], batch_size: int = 128
    ) -> list[list[float]]:
        self.calls += 1
        return [hashed_embedding(text, self.dimension) for text in texts]


class StubLLM(BaseLLM):
    """Reemplazo de `OpenAILLM` que responde con el contexto que recibió.

    Devuelve el mismo formato JSON que pide el prompt, de modo que el parseo y
    la detección de abstención del harness se ejerciten de verdad. Si el prompt
    llegó sin contexto, se abstiene: así se comprueba que esa rama se puntúa.
    """

    def __init__(self, max_chars: int = 300):
        self.max_chars = max_chars
        self.prompts: list[str] = []

    def _extract_context(self, prompt: str) -> str:
        bloque = prompt.split("Contexto:", 1)

        if len(bloque) < 2:
            return ""

        return bloque[1].split("Pregunta:", 1)[0].strip()

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)

        context = self._extract_context(prompt)

        if not context:
            answer = "NO_SE: no tengo suficiente información en el contexto."
            sources: list[str] = []
        else:
            answer = context[: self.max_chars]
            sources = [line for line in context.splitlines() if line][:3]

        return json.dumps(
            {"answer": answer, "sources": sources}, ensure_ascii=False
        )
