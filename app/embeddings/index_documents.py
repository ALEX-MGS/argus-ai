import asyncio
from app.embeddings.vector_store import VectorStore
from app.embeddings.embedding_service import EmbeddingService
from app.processing.text_splitter import split_text


documents = [
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
async def main():

    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    vectors = []

    for doc in documents:
        chunks = split_text(doc)

        for chunk in chunks:
            embedding = await embedding_service.embed(chunk)
            vector_store.add(embedding, chunk)

    for doc, vector in zip(documents, vectors):
        vector_store.add(vector, doc)
    vector_store.save()

    print("Índice creado y guardado")

asyncio.run(main())
