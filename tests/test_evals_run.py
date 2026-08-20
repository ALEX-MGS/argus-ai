"""Test del harness de punta a punta, con dobles y sin API.

`count_tokens` se sustituye porque tiktoken descarga su tabla de codificación la
primera vez. Aquí interesa la lógica del harness, no el tokenizador, que ya
tiene sus propios tests.
"""

import json

import pytest

from app.embeddings.vector_store import VectorStore
from app.pipeline import RAGPipeline
from evals import run as run_module
from evals.dataset import EvalCase
from evals.scoring import summarize
from evals.stubs import StubEmbeddingService, StubLLM, hashed_embedding
from tests.test_pipeline import run


DIMENSION = 1536


@pytest.fixture(autouse=True)
def sin_tokenizador(monkeypatch):
    monkeypatch.setattr(run_module, "count_tokens", lambda texto: len(texto.split()))


@pytest.fixture
def pipeline() -> RAGPipeline:
    store = VectorStore(dimension=DIMENSION)

    documentos = [
        ("FAISS reporta la distancia L2 al cuadrado, evitando la raíz.", "metric.md"),
        ("Los índices de FAISS se almacenan en RAM.", "guidelines.md"),
        ("El parámetro nprobe controla cuántas celdas se visitan.", "faster.md"),
    ]

    for texto, fuente in documentos:
        store.add(hashed_embedding(texto, DIMENSION), texto, source=fuente)

    return RAGPipeline(
        embedding_service=StubEmbeddingService(DIMENSION),
        vector_store=store,
        llm=StubLLM(),
    )


def _caso(**kwargs) -> EvalCase:
    base = {
        "id": "f01",
        "type": "factual",
        "question": "¿Qué distancia reporta FAISS con METRIC_L2?",
        "expected": "al cuadrado",
        "expected_any": ["al cuadrado"],
        "must_cite": ["metric.md"],
    }
    base.update(kwargs)
    return EvalCase(**base)


def test_evalua_un_caso_y_lo_puntua(pipeline):
    resultado = run(run_module.evaluate_case(pipeline, _caso()))

    assert resultado.case.id == "f01"
    assert resultado.answer
    assert resultado.prompt_tokens > 0
    assert resultado.retrieved_sources
    # El doble de LLM devuelve el contexto que recibió, así que si el documento
    # correcto llegó al prompt, la respuesta contiene el hecho esperado.
    assert resultado.correct is (resultado.recall_context == 1.0)


def test_recall_de_contexto_nunca_supera_al_de_recuperados(pipeline):
    """El contexto es un subconjunto de lo recuperado; lo contrario sería un bug."""
    resultado = run(run_module.evaluate_case(pipeline, _caso()))

    assert resultado.recall_context <= resultado.recall_retrieved


def test_caso_de_ausencia_con_contexto_vacio_se_puntua_como_abstencion(pipeline):
    """Con un umbral que filtra todo, el doble se abstiene y eso es correcto."""
    pipeline.threshold = 0.0001

    caso = _caso(type="ausencia", expected="NO_SE", expected_any=[], must_cite=[])

    resultado = run(run_module.evaluate_case(pipeline, caso))

    assert resultado.abstained
    assert resultado.correct
    assert resultado.recall_retrieved is None


def test_reporte_completo_se_escribe_y_es_json_valido(pipeline, tmp_path, capsys):
    casos = [
        _caso(),
        _caso(id="t01", type="trampa", question="¿Kubernetes?",
              expected="NO_SE", expected_any=[], must_cite=[]),
    ]

    resultados = [run(run_module.evaluate_case(pipeline, c)) for c in casos]
    resumen = summarize(resultados)

    run_module.print_report(resultados, resumen)

    destino = tmp_path / "reporte.json"
    run_module.write_report(destino, resumen, resultados, config={"stub": True})

    payload = json.loads(destino.read_text(encoding="utf-8"))

    assert payload["summary"]["total"] == 2
    assert len(payload["cases"]) == 2
    assert payload["config"]["stub"] is True
    assert "timestamp" in payload

    # El reporte impreso debe salir sin reventar y traer las métricas clave.
    salida = capsys.readouterr().out
    assert "Precisión global" in salida
    assert "Tasa de alucinación" in salida


def test_parse_threshold_acepta_none():
    assert run_module.parse_threshold("none") is None
    assert run_module.parse_threshold("2.0") == 2.0
