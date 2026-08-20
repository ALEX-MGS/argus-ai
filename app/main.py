"""Chat por línea de comandos sobre el corpus indexado.

Requiere que el índice ya exista. Para construirlo:

    python -m app.embeddings.index_documents <carpeta_del_corpus>
"""

import asyncio

from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.vector_store import VectorStore
from app.models.openai_llm import OpenAILLM
from app.pipeline import RAGPipeline


async def main() -> None:
    vector_store = VectorStore()
    vector_store.load()

    pipeline = RAGPipeline(
        embedding_service=EmbeddingService(),
        vector_store=vector_store,
        llm=OpenAILLM(),
    )

    # El historial vive aquí, no en el pipeline: la CLI es quien tiene una
    # conversación. La evaluación llama al mismo pipeline sin historial.
    history: list[str] = []

    while True:
        query = input("Haz una pregunta (o escribe salir): ")

        if query.lower() == "salir":
            break

        result = await pipeline.answer(query, history)

        history.append(f"Usuario: {query}")
        history.append(f"Asistente: {result.answer}")
        history = history[-pipeline.history_entries :]

        print("\nRespuesta final:\n")
        print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
