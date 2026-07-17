import asyncio
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.vector_store import VectorStore
from app.models.openai_llm import OpenAILLM


def rerank(query, docs):
    scored_docs = []

    for doc in docs:
        text = doc["text"]

        score = sum(1 for word in query.lower().split() if word in text.lower())

        scored_docs.append((score, doc))

    scored_docs.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scored_docs]

async def main():

    # 1️⃣ inicializaciones
    embedding_service = EmbeddingService()
    llm = OpenAILLM()

    vector_store = VectorStore()
    vector_store.load()

    # 2️⃣ memoria de conversación
    chat_history = []

    # 3️⃣ loop del chat
    while True:

        query = input("Haz una pregunta (o escribe salir): ")

        if query.lower() == "salir":
            break

        # guardar pregunta
        chat_history.append(f"Usuario: {query}")

        # limitar historial
        chat_history = chat_history[-6:]

        # embeddings
        query_vector = await embedding_service.embed(query)

        # retrieval
        retrieved_context = vector_store.search(
            query_vector,
            k=10,
            threshold=2.0
        )

        # rerank
        reranked_docs = rerank(query, retrieved_context)

        # top docs
        top_docs = reranked_docs[:3]

        # contexto
        context_text = "\n".join(
            [doc["text"] for doc in top_docs]
        )

        # prompt
        prompt = f"""
Historial:
{chr(10).join(chat_history)}

Contexto:
{context_text}

Pregunta:
{query}

Responde analizando la información.
No copies literalmente el contexto.
"""

        # respuesta
        response = await llm.generate(prompt)

        # guardar respuesta
        chat_history.append(f"Asistente: {response}")

        # limitar otra vez
        chat_history = chat_history[-6:]

        # imprimir
        print("\nRespuesta final:\n")
        print(response)


if __name__ == "__main__":
    asyncio.run(main())