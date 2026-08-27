"""Tests del reordenamiento léxico. Sin llamadas a API.

Cada test corresponde a un defecto medido en la línea base: el rerank anterior
bajaba el recall@3 de 0.739 a 0.609 con preguntas en español sobre un corpus en
inglés, porque las palabras vacías coincidían dentro de palabras ajenas.
"""

from __future__ import annotations

from app.pipeline import content_terms, rerank


def doc(text: str, source: str) -> dict:
    return {"text": text, "source": source}


def orden(docs: list[dict]) -> list[str]:
    return [d["source"] for d in docs]


def test_no_coincide_dentro_de_otra_palabra():
    """'de' no debe encontrar coincidencia en 'index' ni en 'order'."""
    assert "de" not in content_terms("index order under")


def test_descarta_palabras_vacias():
    terminos = content_terms("¿Qué devuelve el parámetro nprobe de FAISS?")

    assert "nprobe" in terminos
    assert "faiss" in terminos
    assert "que" not in terminos
    assert "el" not in terminos
    assert "de" not in terminos


def test_sube_el_documento_con_el_termino_tecnico():
    docs = [
        doc("Generic text about vectors and search.", "ruido.md"),
        doc("The nprobe parameter adjusts the speed accuracy tradeoff.", "correcto.md"),
    ]

    assert orden(rerank("What does nprobe do?", docs))[0] == "correcto.md"


def test_documento_largo_no_gana_por_repetir():
    """Se cuentan términos distintos, no ocurrencias."""
    docs = [
        doc("index index index index index index index index", "repetitivo.md"),
        doc("index nprobe", "preciso.md"),
    ]

    assert orden(rerank("index nprobe", docs))[0] == "preciso.md"


def test_sin_senal_lexica_se_conserva_el_orden_de_faiss():
    """El caso español-sobre-inglés: si nada coincide, no se reordena.

    Es la corrección central: antes el ruido de las palabras vacías reordenaba
    y tiraba fuera del top-3 documentos que FAISS había encontrado bien.
    """
    docs = [
        doc("Faiss reports squared Euclidean distance.", "primero.md"),
        doc("The index factory interprets a string.", "segundo.md"),
        doc("Guidelines to choose an index.", "tercero.md"),
    ]

    pregunta = "¿Cuál es la dimensionalidad máxima soportada?"

    assert orden(rerank(pregunta, docs)) == ["primero.md", "segundo.md", "tercero.md"]


def test_empates_conservan_el_orden_de_entrada():
    docs = [doc("nprobe", f"{i}.md") for i in range(5)]

    assert orden(rerank("nprobe", docs)) == [f"{i}.md" for i in range(5)]


def test_lista_vacia():
    assert rerank("cualquier cosa", []) == []


def test_terminos_de_una_letra_se_ignoran():
    """Evita que variables sueltas como 'd' o 'M' dominen el puntaje."""
    assert content_terms("d M x") == set()
