"""import asyncio
from app.models.openai_llm import OpenAILLM
import logging
from app.core.logging_config import setup_logging


async def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    llm = OpenAILLM()

    prompt = "Explica brevemente qué es un sistema RAG."

    logger.info("Enviando prompt al modelo...")

    response = await llm.generate(prompt)

    logger.info("Respuesta recibida correctamente.")

    print("\nRespuesta del modelo:\n")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
"""
"""

import asyncio

from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.vector_store import VectorStore


async def main():
    embedding_service = EmbeddingService()

    # Textos de prueba
    texts = [
        "Python es un lenguaje de programación",
        "La inteligencia artificial está cambiando el mundo",
        "Los muebles de MDF son muy usados en carpintería"
    ]

    # 1️⃣ Generar embedding del primer texto para saber dimensión
    first_vector = await embedding_service.embed(texts[0])
    dimension = len(first_vector)

    # 2️⃣ Crear vector store
    vector_store = VectorStore(dimension)

    # 3️⃣ Agregar todos los textos
    for text in texts:
        vector = await embedding_service.embed(text)
        vector_store.add(vector, text)

    # 4️⃣ Hacer búsqueda
    query = "¿Qué material se usa para hacer muebles?"
    query_vector = await embedding_service.embed(query)

    results = vector_store.search(query_vector, k=2)

    print("\nResultados de búsqueda:")
    for r in results:
        print("-", r)


if __name__ == "__main__":
    asyncio.run(main())
"""

import asyncio

from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.vector_store import VectorStore
from app.models.openai_llm import OpenAILLM


async def main():
    embedding_service = EmbeddingService()
    llm = OpenAILLM()

    texts = [
"Python es un lenguaje de programación muy usado en ciencia de datos.",
"La inteligencia artificial está transformando muchas industrias.",
"Los muebles de MDF son muy usados en carpintería moderna.",
"El MDF es un tablero fabricado con fibras de madera prensadas.",
"La madera maciza es un material tradicional en carpintería.",
"Los gabinetes de cocina suelen fabricarse con MDF o triplay.",
"La carpintería utiliza herramientas como sierras, taladros y lijadoras.",
"FAISS es una biblioteca para búsqueda eficiente de vectores.",
"Los embeddings convierten texto en vectores numéricos.",
"La búsqueda semántica permite encontrar información por significado.",
"Los modelos de lenguaje pueden generar texto automáticamente.",
"Un sistema RAG combina búsqueda de información con generación de texto.",
"Los talleres de carpintería producen muebles personalizados.",
"Las cocinas modernas utilizan gabinetes modulares.",
"El barniz protege la madera contra la humedad.",
"Los sistemas de inteligencia artificial utilizan grandes cantidades de datos.",
"Los modelos de lenguaje grandes se entrenan con millones de textos.",
"La recuperación de información es una parte importante de muchos sistemas de IA.",
"Las bases de datos vectoriales almacenan representaciones numéricas de documentos.",
"Los agentes de inteligencia artificial pueden automatizar tareas complejas."
    ]

    # Crear índice
    vector_store = VectorStore()
    vector_store.load()

    # Pregunta del usuario
    query = input("Haz una pregunta: ")


    # 1️⃣ Buscar contexto relevante
    query_vector = await embedding_service.embed(query)
    

    retrieved_context = vector_store.search(query_vector, k=10, threshold=2.0)

    "impresion para ver lista retieved context y evaluarla"
    print("\nResultados de búsqueda:")
    for doc in retrieved_context:
        print("-", doc)
    top_docs = retrieved_context[:3]

    context_text = "\n".join(top_docs)

    # 2️⃣ Prompt estructurado
    prompt = f"""
Responde la pregunta usando SOLO el contexto proporcionado.

Contexto:
{context_text}

Pregunta:
{query}

Si la respuesta no está en el contexto, di que no tienes suficiente información.
Devuelve la respuesta en JSON con este formato:

{{
 "answer": "respuesta clara",
 "sources": ["fragmentos de contexto utilizados"]
}}

"""


    # 3️⃣ Generar respuesta
    response = await llm.generate(prompt)

    print("\nRespuesta final:\n")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
