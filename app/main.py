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
    #inicializar servicios 
    embedding_service = EmbeddingService()
    llm = OpenAILLM()
    #cargar textos
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
    # historial de chat
    chat_history = []

    # Crear índice
    vector_store = VectorStore(dimension=1536)
    vector_store.load()

    # Pregunta del usuario
    query = input("Haz una pregunta: ")

    chat_history.append(f"Usuario: {query}")
    "limite de acumulacio de historial de historial de contexto"
    chat_history = chat_history[-6:]


    # 1️⃣ Buscar contexto relevante
    query_vector = await embedding_service.embed(query)
    

    retrieved_context = vector_store.search(query_vector, k=10)

    reranked_docs = rerank(query, retrieved_context)


    " top_docs = reranked_docs[:3]"

    "impresion para ver lista retrieved context y evaluarla"
    print("\nResultados de búsqueda:")
    for doc in retrieved_context:
        print("-", doc["text"], "| fuente:", doc["source"])
    top_docs = retrieved_context[:3]

    "impresion para ver lista rerranked y evaluar diferencias con retrived context"
    print("\nResultados de reranked:")
    for doc in reranked_docs:
        print("-", doc["text"], "| fuente:", doc["source"])
    top_reranked = reranked_docs[:3]

    real_sources = list(set([doc["source"] for doc in top_docs]))
    print("\nFuentes reales:")
    for s in real_sources:
        print("-", s)
    context_text = "\n".join([doc["text"] for doc in top_docs])

    sources = list(set([doc["source"] for doc in top_docs]))

    # 2️⃣ Prompt estructurado
    prompt = f"""

    
Responde la pregunta usando SOLO el contexto proporcionado.

Historial:
{chr(10).join(chat_history)}

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

    chat_history.append(f"Asistente: {response}")
    "limite de acumulacion de historial de historial de contexto"
    chat_history = chat_history[-6:]

    print("\nRespuesta final:\n")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())