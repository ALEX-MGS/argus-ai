"""Tests del pipeline de consulta. No requieren API ni tokenizador.

El objetivo principal es dejar fijado el comportamiento que se extrajo de
`main.py`, para que las mejoras de la siguiente fase se midan contra algo
verificable y no contra el recuerdo de cómo funcionaba.
"""

import asyncio

from app.embeddings.vector_store import VectorStore
from app.pipeline import RAGPipeline
from evals.stubs import StubEmbeddingService, StubLLM, hashed_embedding


DIMENSION = 1536


def run(coro):
    """Ejecuta una corrutina desde un test síncrono.

    Evita depender de `pytest-asyncio` para lo poco que se necesita aquí.
    """
    return asyncio.run(coro)


def _store(*docs: tuple[str, str]) -> VectorStore:
    """Índice en memoria a partir de pares (texto, fuente)."""
    store = VectorStore(dimension=DIMENSION)

    for text, source in docs:
        store.add(hashed_embedding(text, DIMENSION), text, source=source)

    return store


def _pipeline(store: VectorStore, **kwargs) -> RAGPipeline:
    return RAGPipeline(
        embedding_service=StubEmbeddingService(DIMENSION),
        vector_store=store,
        llm=StubLLM(),
        **kwargs,
    )


def test_devuelve_respuesta_y_contexto():
    store = _store(
        ("FAISS reporta la distancia L2 al cuadrado.", "metric.md"),
        ("Los índices de FAISS se almacenan en RAM.", "guidelines.md"),
    )

    result = run(_pipeline(store).answer("¿Qué distancia reporta FAISS?"))

    assert result.answer
    assert result.context
    assert all("source" in doc for doc in result.context)


def test_el_contexto_se_corta_en_top_n():
    store = _store(*[(f"documento número {i}", f"doc{i}.md") for i in range(10)])

    result = run(_pipeline(store, top_n=3).answer("documento"))

    assert len(result.context) == 3
    assert len(result.retrieved) > len(result.context)


def test_recuperados_traen_puntaje():
    store = _store(("contenido de prueba", "a.md"))

    result = run(_pipeline(store).answer("contenido"))

    assert all("score" in doc for doc in result.retrieved)


def test_la_busqueda_no_contamina_los_documentos_guardados():
    """El puntaje es de la consulta, no del documento.

    Anotarlo sobre el dict almacenado dejaría el puntaje de una consulta
    pegado al documento y visible en la siguiente.
    """
    store = _store(("contenido de prueba", "a.md"))

    run(_pipeline(store).answer("primera consulta"))

    assert all("score" not in doc for doc in store.documents)


def test_el_prompt_conserva_la_estructura_original():
    store = _store(("un texto cualquiera", "a.md"))

    result = run(_pipeline(store).answer("mi pregunta"))

    for seccion in ("Historial:", "Contexto:", "Pregunta:"):
        assert seccion in result.prompt

    assert '"answer"' in result.prompt
    assert '"sources"' in result.prompt


def test_la_consulta_actual_entra_en_el_historial_del_prompt():
    """Comportamiento heredado: la pregunta aparece dos veces en el prompt.

    Se fija aquí a propósito. Cuando se corrija, este test debe cambiar de
    forma deliberada y no por accidente.
    """
    store = _store(("un texto cualquiera", "a.md"))

    result = run(_pipeline(store).answer("mi pregunta"))

    historial = result.prompt.split("Contexto:")[0]

    assert "Usuario: mi pregunta" in historial


def test_el_historial_se_recorta_al_limite():
    store = _store(("un texto cualquiera", "a.md"))

    historia_larga = [f"Usuario: pregunta {i}" for i in range(20)]

    result = run(
        _pipeline(store, history_entries=6).answer("actual", history=historia_larga)
    )

    historial = result.prompt.split("Contexto:")[0]

    assert "Usuario: pregunta 19" in historial
    assert "Usuario: pregunta 0" not in historial


def test_el_pipeline_no_muta_el_historial_recibido():
    store = _store(("un texto cualquiera", "a.md"))

    historia = ["Usuario: anterior"]

    run(_pipeline(store).answer("actual", history=historia))

    assert historia == ["Usuario: anterior"]


def test_umbral_estricto_deja_el_contexto_vacio():
    """Con un umbral que sí filtra, no debe llegar contexto al prompt."""
    store = _store(("contenido totalmente ajeno", "a.md"))

    result = run(
        _pipeline(store, threshold=0.0001).answer("consulta sin ninguna relación")
    )

    assert result.retrieved == []
    assert result.context == []
