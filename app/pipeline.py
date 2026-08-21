"""Pipeline de recuperación y generación.

Extraído del loop de chat para que el chat interactivo y el arnés de
evaluación ejerciten exactamente el mismo código. Si midiéramos una copia del
pipeline en vez del pipeline real, los números no dirían nada sobre el sistema.

Los valores por defecto reproducen el comportamiento vigente del sistema
(k=10, threshold=2.0, 3 documentos al prompt). No cambiarlos sin medir antes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.vector_store import VectorStore
from app.models.base_llm import BaseLLM


DEFAULT_K = 10
DEFAULT_THRESHOLD = 2.0
DEFAULT_TOP_DOCS = 3
DEFAULT_HISTORY_TURNS = 6


@dataclass
class RetrievedChunk:
    """Un fragmento devuelto por el índice."""

    text: str
    source: str


@dataclass
class PipelineResult:
    """Resultado completo de una consulta, con las señales intermedias.

    Guarda tanto lo recuperado como lo que efectivamente llegó al prompt, para
    poder distinguir un fallo de recuperación de uno de generación.
    """

    query: str
    answer: str
    prompt: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    sent_to_prompt: list[RetrievedChunk] = field(default_factory=list)

    @property
    def retrieved_sources(self) -> list[str]:
        return [chunk.source for chunk in self.retrieved]

    @property
    def prompt_sources(self) -> list[str]:
        return [chunk.source for chunk in self.sent_to_prompt]


def rerank(query: str, docs: list[dict]) -> list[dict]:
    """Reordena por coincidencia léxica de palabras de la consulta.

    Limitación conocida (punto 2.3 del plan): cuenta coincidencias por
    substring y no descarta stopwords, así que premia documentos largos.
    Se conserva tal cual para que la línea base mida el sistema real.
    """
    scored_docs = []

    for doc in docs:
        text = doc["text"]
        score = sum(1 for word in query.lower().split() if word in text.lower())
        scored_docs.append((score, doc))

    scored_docs.sort(reverse=True, key=lambda item: item[0])

    return [doc for _, doc in scored_docs]


def build_prompt(query: str, context_text: str, chat_history: list[str]) -> str:
    """Arma el prompt. Idéntico al que usaba el loop de chat."""
    return f"""
Historial:
{chr(10).join(chat_history)}

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


class RagPipeline:
    """Orquesta embedding, recuperación, rerank y generación."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm: BaseLLM,
        k: int = DEFAULT_K,
        threshold: float | None = DEFAULT_THRESHOLD,
        top_docs: int = DEFAULT_TOP_DOCS,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.threshold = threshold
        self.top_docs = top_docs

    async def answer(
        self, query: str, chat_history: list[str] | None = None
    ) -> PipelineResult:
        """Responde una consulta y devuelve el resultado con sus intermedios."""
        chat_history = chat_history or []

        query_vector = await self.embedding_service.embed(query)

        retrieved_raw = self.vector_store.search(
            query_vector, k=self.k, threshold=self.threshold
        )

        reranked = rerank(query, retrieved_raw)
        top = reranked[: self.top_docs]

        context_text = "\n".join(doc["text"] for doc in top)
        prompt = build_prompt(query, context_text, chat_history)

        response = await self.llm.generate(prompt)

        to_chunks = lambda docs: [  # noqa: E731
            RetrievedChunk(text=d["text"], source=d.get("source", "unknown"))
            for d in docs
        ]

        return PipelineResult(
            query=query,
            answer=response,
            prompt=prompt,
            retrieved=to_chunks(reranked),
            sent_to_prompt=to_chunks(top),
        )


def build_default_pipeline() -> RagPipeline:
    """Construye el pipeline con el índice ya persistido en disco."""
    from app.models.openai_llm import OpenAILLM

    vector_store = VectorStore()
    vector_store.load()

    return RagPipeline(
        embedding_service=EmbeddingService(),
        vector_store=vector_store,
        llm=OpenAILLM(),
    )
