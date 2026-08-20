"""Reordenamiento de los documentos recuperados.

Implementación léxica actual: cuenta cuántas palabras de la consulta aparecen
dentro del texto del documento. Tiene limitaciones conocidas —cuenta stopwords
y hace coincidencia de subcadena, no de palabra completa— documentadas en
`docs/ARGUS-plan.md` §2.3. Se conserva tal cual para que la línea base mida el
sistema como está hoy.
"""

from __future__ import annotations


def rerank(query: str, docs: list[dict]) -> list[dict]:
    """Ordena los documentos por coincidencia léxica con la consulta.

    El orden es descendente por puntaje. `list.sort` es estable incluso con
    `reverse=True`, así que los empates conservan el orden en que los devolvió
    el índice vectorial. Eso importa: con consultas donde ninguna palabra
    coincide, el resultado es exactamente el orden por similitud.
    """
    scored_docs = []

    for doc in docs:
        text = doc["text"]

        score = sum(1 for word in query.lower().split() if word in text.lower())

        scored_docs.append((score, doc))

    scored_docs.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scored_docs]
