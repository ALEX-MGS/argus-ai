"""Tests de la puntuación de respuestas."""

from evals.dataset import EvalCase
from evals.scoring import (
    CaseResult,
    answer_is_correct,
    citation_recall,
    is_abstention,
    normalize,
    parse_answer,
    summarize,
)


def _caso(**kwargs) -> EvalCase:
    base = {
        "id": "x1",
        "type": "factual",
        "question": "¿pregunta?",
        "expected": "respuesta",
        "expected_any": ["al cuadrado"],
        "must_cite": ["a.md"],
    }
    base.update(kwargs)
    return EvalCase(**base)


def _resultado(caso: EvalCase, **kwargs) -> CaseResult:
    base = {
        "case": caso,
        "answer": "",
        "sources_cited": [],
        "recall_retrieved": None,
        "recall_context": None,
        "correct": False,
        "abstained": False,
        "prompt_tokens": 0,
    }
    base.update(kwargs)
    return CaseResult(**base)


# --- normalización ---------------------------------------------------------


def test_normalize_quita_acentos_y_mayusculas():
    assert normalize("Distancia AL CUADRADO ó más") == "distancia al cuadrado o mas"


# --- parseo de la respuesta ------------------------------------------------


def test_parse_answer_json_plano():
    crudo = '{"answer": "es al cuadrado", "sources": ["a.md"]}'

    assert parse_answer(crudo) == ("es al cuadrado", ["a.md"])


def test_parse_answer_con_cerca_de_markdown():
    crudo = '```json\n{"answer": "hola", "sources": []}\n```'

    answer, sources = parse_answer(crudo)

    assert answer == "hola"
    assert sources == []


def test_parse_answer_con_prosa_alrededor():
    crudo = 'Claro:\n{"answer": "hola", "sources": ["a.md"]}\nEspero que sirva.'

    assert parse_answer(crudo)[0] == "hola"


def test_parse_answer_json_invalido_devuelve_el_crudo():
    """Un fallo de formato no debe contarse como respuesta equivocada."""
    crudo = "{esto no es json pero dice al cuadrado}"

    answer, sources = parse_answer(crudo)

    assert answer == crudo
    assert sources == []


def test_parse_answer_sin_json():
    crudo = "La distancia es al cuadrado."

    assert parse_answer(crudo) == (crudo, [])


def test_parse_answer_sources_no_lista():
    crudo = '{"answer": "hola", "sources": "a.md"}'

    assert parse_answer(crudo)[1] == ["a.md"]


# --- abstención ------------------------------------------------------------


def test_detecta_abstencion_en_varias_formas():
    for texto in (
        "NO_SE",
        "No tengo suficiente información en el contexto.",
        "El contexto no especifica ese valor.",
        "Esa información no aparece en el contexto proporcionado.",
    ):
        assert is_abstention(texto), texto


def test_una_respuesta_normal_no_es_abstencion():
    assert not is_abstention("FAISS devuelve la distancia al cuadrado.")


# --- corrección de la respuesta -------------------------------------------


def test_acierto_por_palabra_esperada():
    assert answer_is_correct(_caso(), "Devuelve la distancia al cuadrado")


def test_acierto_ignora_acentos_y_mayusculas():
    assert answer_is_correct(_caso(expected_any=["raíz cuadrada"]), "La RAIZ CUADRADA")


def test_fallo_si_no_aparece_lo_esperado():
    assert not answer_is_correct(_caso(), "Devuelve la distancia euclidiana simple")


def test_abstenerse_cuando_si_habia_respuesta_es_fallo():
    """Aunque la disculpa mencione de pasada la palabra clave."""
    caso = _caso()
    respuesta = "No tengo suficiente información sobre la distancia al cuadrado."

    assert not answer_is_correct(caso, respuesta)


def test_abstencion_correcta_en_caso_de_ausencia():
    caso = _caso(type="ausencia", expected="NO_SE", expected_any=[], must_cite=[])

    assert answer_is_correct(caso, "El contexto no especifica ese valor.")


def test_responder_en_caso_trampa_es_fallo():
    caso = _caso(type="trampa", expected="NO_SE", expected_any=[], must_cite=[])

    assert not answer_is_correct(caso, "Se recomienda usar el modelo X.")


# --- recall de citas -------------------------------------------------------


def test_recall_completo():
    caso = _caso(must_cite=["a.md", "b.md"])
    docs = [{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}]

    assert citation_recall(caso, docs) == 1.0


def test_recall_parcial_en_multi_hop():
    caso = _caso(must_cite=["a.md", "b.md"])
    docs = [{"source": "a.md"}, {"source": "c.md"}]

    assert citation_recall(caso, docs) == 0.5


def test_recall_es_none_si_no_se_espera_cita():
    caso = _caso(must_cite=[])

    assert citation_recall(caso, [{"source": "a.md"}]) is None


# --- agregación ------------------------------------------------------------


def test_summarize_separa_por_tipo_y_calcula_alucinacion():
    factual = _caso(id="f1", type="factual")
    ausencia = _caso(id="a1", type="ausencia", expected_any=[], must_cite=[])
    trampa = _caso(id="t1", type="trampa", expected_any=[], must_cite=[])

    resultados = [
        _resultado(factual, correct=True, recall_retrieved=1.0, recall_context=0.5,
                   prompt_tokens=100),
        _resultado(ausencia, correct=True, abstained=True, prompt_tokens=100),
        # Respondió cuando debía abstenerse: alucinación.
        _resultado(trampa, correct=False, abstained=False, prompt_tokens=100),
    ]

    resumen = summarize(resultados)

    assert resumen["total"] == 3
    assert resumen["by_type"]["factual"]["accuracy"] == 1.0
    assert resumen["hallucination_rate"] == 0.5
    assert resumen["hallucinated_ids"] == ["t1"]
    assert resumen["recall_retrieved"] == 1.0
    assert resumen["recall_context"] == 0.5
    assert resumen["avg_output_tokens"] is None


def test_summarize_sin_casos_de_abstencion():
    resultados = [_resultado(_caso(id="f1"), correct=True)]

    assert summarize(resultados)["hallucination_rate"] is None
