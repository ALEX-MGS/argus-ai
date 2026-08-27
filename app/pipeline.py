"""Pipeline de recuperación y generación.

Extraído del loop de chat para que el chat interactivo y el arnés de
evaluación ejerciten exactamente el mismo código. Si midiéramos una copia del
pipeline en vez del pipeline real, los números no dirían nada sobre el sistema.

Los valores por defecto reproducen el comportamiento vigente del sistema
(k=10, threshold=2.0, 3 documentos al prompt). No cambiarlos sin medir antes.
"""

from __future__ import annotations

import re
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


# Palabras vacías: no aportan señal y, al buscarse por substring, encontraban
# coincidencias dentro de otras palabras ("de" dentro de "index", "order").
STOPWORDS = frozenset("""
a al algo alguna algunas alguno algunos ante antes como con contra cual cuales
cuando de del desde donde dos e el ella ellas ello ellos en entre era es esa
esas ese eso esos esta estan estas este esto estos ha hace hacer hasta hay la
las le les lo los mas me mi mis mucho muy no nos o os otra otras otro otros
para pero poco por porque que quien se ser si sin sobre son su sus tambien
tan tanto te tiene todo todos tu tus un una uno unos y ya
a about all also an and any are as at be been but by can could do does for
from had has have how i if in into is it its may more most no not of on once
only or other our out over should so some such than that the their them then
there these they this those through to too under until up use used using was
we were what when where which while who why will with would you your
""".split())

_WORD = re.compile(r"\w+", re.UNICODE)


def content_terms(text: str) -> set[str]:
    """Palabras con contenido de un texto: completas, en minúscula, sin vacías."""
    return {
        palabra
        for palabra in _WORD.findall(text.lower())
        if palabra not in STOPWORDS and len(palabra) > 1
    }


def rerank(query: str, docs: list[dict]) -> list[dict]:
    """Reordena por cuántos términos con contenido de la consulta aparecen.

    Tres decisiones, cada una contra un defecto medido de la versión anterior:

    - Palabras completas, no substrings: "de" ya no coincide dentro de "index".
    - Sin palabras vacías: solo pesan los términos que discriminan.
    - Términos distintos, no ocurrencias: un documento largo no puede subir por
      repetir la misma palabra muchas veces.

    Si la consulta no comparte ningún término con ningún documento —el caso de
    preguntar en español sobre un corpus en inglés— todos empatan en cero y el
    orden de FAISS se conserva, porque `sort` es estable. El rerank deja de
    aportar, pero tampoco estorba.
    """
    terminos_consulta = content_terms(query)

    scored_docs = [
        (len(terminos_consulta & content_terms(doc["text"])), doc) for doc in docs
    ]

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
Responde en el mismo idioma en que está escrita la pregunta.

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
        use_rerank: bool = True,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.threshold = threshold
        self.top_docs = top_docs
        self.use_rerank = use_rerank

    async def retrieve(self, query: str) -> tuple[list[dict], list[dict]]:
        """Recupera del índice y devuelve (todos_ordenados, los_que_van_al_prompt).

        Con `use_rerank=False` se conserva el orden de FAISS, que ordena por
        distancia. Sirve para medir si el rerank léxico aporta o estorba.
        """
        query_vector = await self.embedding_service.embed(query)

        retrieved_raw = self.vector_store.search(
            query_vector, k=self.k, threshold=self.threshold
        )

        ordenados = rerank(query, retrieved_raw) if self.use_rerank else retrieved_raw

        return ordenados, ordenados[: self.top_docs]

    async def answer(
        self, query: str, chat_history: list[str] | None = None
    ) -> PipelineResult:
        """Responde una consulta y devuelve el resultado con sus intermedios."""
        chat_history = chat_history or []

        reranked, top = await self.retrieve(query)

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


def build_default_pipeline(use_rerank: bool = True) -> RagPipeline:
    """Construye el pipeline con el índice ya persistido en disco."""
    from app.models.openai_llm import OpenAILLM

    vector_store = VectorStore()
    vector_store.load()

    return RagPipeline(
        embedding_service=EmbeddingService(),
        vector_store=vector_store,
        llm=OpenAILLM(),
        use_rerank=use_rerank,
    )
