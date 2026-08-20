"""Pipeline de consulta: de una pregunta a una respuesta fundamentada.

Extraído de `main.py` sin alterar el comportamiento, para que tanto la CLI como
el harness de evaluación entren por el mismo camino. Sin esto, medir el sistema
exigiría replicar la lógica dentro del harness, y se acabaría midiendo una copia
en vez del sistema real.

El pipeline es **sin estado**: recibe el historial y lo devuelve intacto. Quien
llama decide qué conservar entre turnos.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.vector_store import VectorStore
from app.models.base_llm import BaseLLM
from app.retrieval.rerank import rerank


# Valores que hoy están fijos en el código. Se exponen como parámetros para que
# la evaluación pueda barrerlos sin editar el pipeline.
DEFAULT_K = 10
DEFAULT_TOP_N = 3
DEFAULT_HISTORY_ENTRIES = 6

# El umbral no filtra en la práctica: `IndexFlatL2` devuelve distancia L2 al
# cuadrado, cuyo rango con embeddings normalizados es [0, 4], así que 2.0 solo
# descarta similitud coseno negativa (ver `docs/ARGUS-plan.md` §2.1). Se conserva
# el valor actual a propósito: la línea base debe medir el sistema como está.
DEFAULT_THRESHOLD = 2.0


@dataclass
class PipelineResult:
    """Todo lo observable de una consulta, para responder y para medir."""

    query: str

    # Texto crudo devuelto por el modelo. No se parsea aquí: hoy el modelo
    # devuelve un JSON dentro de un string y quien consume decide qué hacer.
    answer: str

    # Lo que devolvió el índice, con su puntaje. Sirve para medir Recall@k.
    retrieved: list[dict]

    # El subconjunto que efectivamente entró al prompt. La diferencia entre
    # ambas listas es lo que el rerank descartó — separarlas permite saber si
    # un fallo vino de la recuperación o del reordenamiento.
    context: list[dict]

    # El prompt final, para contabilizar tokens sin reconstruirlo.
    prompt: str


class RAGPipeline:
    """Orquesta recuperación y generación para una consulta."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm: BaseLLM,
        k: int = DEFAULT_K,
        threshold: float | None = DEFAULT_THRESHOLD,
        top_n: int = DEFAULT_TOP_N,
        history_entries: int = DEFAULT_HISTORY_ENTRIES,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.threshold = threshold
        self.top_n = top_n
        self.history_entries = history_entries

    def build_prompt(
        self, history: list[str], context_text: str, query: str
    ) -> str:
        """Arma el prompt.

        Nota de comportamiento actual: la consulta aparece dos veces —al final
        del historial y como «Pregunta»—, porque el historial se arma incluyendo
        el turno en curso. Se conserva para no alterar la línea base.
        """
        return f"""
Historial:
{chr(10).join(history)}

Contexto:
{context_text}

Pregunta:
{query}

Responde la pregunta usando SOLO el contexto proporcionado.
No copies literalmente el contexto.
Si la respuesta no está en el contexto, di que no tienes suficiente información.

Devuelve la respuesta en JSON con este formato:

{{
 "answer": "respuesta clara",
 "sources": ["fragmentos de contexto utilizados"]
}}
"""

    async def answer(
        self, query: str, history: list[str] | None = None
    ) -> PipelineResult:
        """Responde una consulta con el contexto recuperado del índice.

        Args:
            query: pregunta del usuario.
            history: turnos previos ya formateados (`"Usuario: ..."` /
                `"Asistente: ..."`), sin incluir la consulta actual.

        Returns:
            El resultado con la respuesta y todo lo recuperado en el camino.
        """
        turn_history = list(history or [])
        turn_history.append(f"Usuario: {query}")
        turn_history = turn_history[-self.history_entries :]

        query_vector = await self.embedding_service.embed(query)

        retrieved = self.vector_store.search(
            query_vector, k=self.k, threshold=self.threshold
        )

        context = rerank(query, retrieved)[: self.top_n]

        context_text = "\n".join(doc["text"] for doc in context)

        prompt = self.build_prompt(turn_history, context_text, query)

        answer = await self.llm.generate(prompt)

        return PipelineResult(
            query=query,
            answer=answer,
            retrieved=retrieved,
            context=context,
            prompt=prompt,
        )
