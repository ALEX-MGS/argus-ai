"""Tests del arnés de evaluación. Sin llamadas a API.

Se prueba la calificación, que es donde un error silencioso haría que todas las
métricas posteriores mientan.
"""

from __future__ import annotations

import json

import pytest

from evals.run_eval import (
    EvalCase,
    evaluate_case,
    extract_answer_text,
    is_abstention,
    load_cases,
    matches_expected,
    summarize,
)


# --- extracción de la respuesta ---------------------------------------------

def test_extrae_answer_de_json_valido():
    crudo = json.dumps({"answer": "squared L2 distance", "sources": ["a"]})
    assert extract_answer_text(crudo) == "squared L2 distance"


def test_extrae_answer_de_json_en_bloque_de_codigo():
    crudo = '```json\n{"answer": "nprobe", "sources": []}\n```'
    assert extract_answer_text(crudo) == "nprobe"


def test_json_invalido_devuelve_texto_crudo():
    crudo = "El índice guarda todo en RAM, sin JSON."
    assert extract_answer_text(crudo) == crudo


def test_json_sin_campo_answer_devuelve_crudo():
    crudo = json.dumps({"respuesta": "otra clave"})
    assert extract_answer_text(crudo) == crudo


# --- calificación ------------------------------------------------------------

def test_detecta_abstencion_en_ambos_idiomas():
    assert is_abstention("No tengo suficiente información para responder.")
    assert is_abstention("There is not enough information in the context.")
    assert not is_abstention("IndexFlatL2 devuelve la distancia al cuadrado.")


def test_detecta_abstencion_con_redacciones_distintas():
    """Casos reales de corridas anteriores que la lista literal no atrapaba."""
    assert is_abstention(
        "Faiss maneja vectores de dimensión fija. Sin embargo, el contexto no "
        "especifica un límite máximo exacto para la dimensionalidad."
    )
    assert is_abstention("El wiki no menciona la versión en que se introdujo.")
    assert is_abstention("The documentation does not state a default value.")
    assert is_abstention("That cannot be determined from the provided context.")


def test_no_confunde_una_respuesta_negativa_con_abstencion():
    """Responder 'no' a una pregunta de sí/no no es abstenerse."""
    assert not is_abstention("No, Faiss no soporta ids de tipo string.")
    assert not is_abstention("No, re-training an index is not supported.")


def test_match_por_substring_ignora_mayusculas():
    assert matches_expected("Devuelve la distancia CUADRADA", ["cuadrada"])
    assert not matches_expected("Devuelve la distancia L2", ["cuadrada"])


def test_match_exige_empezar_en_frontera_de_palabra():
    """El esperado no puede caer en medio de otra palabra."""
    assert not matches_expected("Los índices viven en la programación", ["ram"])
    assert matches_expected("Los índices viven en RAM", ["ram"])


def test_match_admite_prefijos_para_cubrir_flexiones():
    """Escribir 'normaliz' debe cubrir normaliza/normalizes/normalizados."""
    assert matches_expected("La transformación normaliza los vectores", ["normaliz"])
    assert matches_expected("It normalizes the input vectors", ["normaliz"])
    assert matches_expected("vectores ya normalizados", ["normaliz"])


def test_match_admite_expresiones_con_simbolos():
    assert matches_expected("los ids faltantes se rellenan con -1", ["-1"])
    assert matches_expected("la relación es 2 - 2 <x,y>", ["2 - 2"])
    assert matches_expected("usa d*4+m*2*4 bytes", ["d*4+m*2*4"])


def test_match_ignora_el_enfasis_de_markdown():
    """El modelo a veces copia el formato del wiki: (_d_ * 4 + _M_ * 2 * 4)."""
    respuesta = "La memoria es (_d_ * 4 + _M_ * 2 * 4) bytes por vector."

    assert matches_expected(respuesta, ["d * 4 + m * 2 * 4"])


def test_caso_factual_correcto():
    case = EvalCase(
        id="q01", type="factual", question="?",
        must_cite=["MetricType-and-distances.md"], expected_any=["squared", "cuadrado"],
    )

    resultado = evaluate_case(
        case,
        "Reporta la distancia euclidiana al cuadrado.",
        prompt_sources=["MetricType-and-distances.md", "FAQ.md"],
        retrieved_sources=["MetricType-and-distances.md", "FAQ.md"],
    )

    assert resultado.answer_correct
    assert resultado.recall_at_prompt
    assert resultado.recall_at_k


def test_fallo_de_recuperacion_se_distingue_del_de_generacion():
    """Si el documento esperado nunca llegó, el fallo es del retriever."""
    case = EvalCase(
        id="q01", type="factual", question="?",
        must_cite=["MetricType-and-distances.md"], expected_any=["cuadrado"],
    )

    resultado = evaluate_case(
        case,
        "Devuelve la distancia L2.",
        prompt_sources=["Faiss-indexes.md"],
        retrieved_sources=["Faiss-indexes.md", "FAQ.md"],
    )

    assert not resultado.answer_correct
    assert not resultado.recall_at_prompt
    assert not resultado.recall_at_k


def test_trampa_correcta_solo_si_se_abstiene():
    case = EvalCase(
        id="q10", type="trampa", question="?", must_cite=[], expected_any=[],
    )

    bien = evaluate_case(case, "No tengo suficiente información.", [], [])
    mal = evaluate_case(case, "Usa un StatefulSet con tres réplicas.", [], [])

    assert bien.answer_correct
    assert not mal.answer_correct


def test_caso_sin_must_cite_no_reporta_recall():
    case = EvalCase(id="q09", type="ausencia", question="?", must_cite=[], expected_any=[])

    resultado = evaluate_case(case, "No tengo suficiente información.", [], [])

    assert resultado.recall_at_prompt is None
    assert resultado.recall_at_k is None


# --- agregación --------------------------------------------------------------

def test_resumen_calcula_las_cuatro_metricas():
    case_ok = EvalCase("q1", "factual", "?", ["A.md"], ["x"])
    case_mal = EvalCase("q2", "factual", "?", ["B.md"], ["y"])
    case_trampa = EvalCase("q3", "trampa", "?", [], [])

    outcomes = [
        evaluate_case(case_ok, "contiene x", ["A.md"], ["A.md"]),
        evaluate_case(case_mal, "no contiene nada", ["Z.md"], ["Z.md"]),
        evaluate_case(case_trampa, "inventé una respuesta", [], []),
    ]

    resumen = summarize(outcomes)

    assert resumen["casos"] == 3
    assert resumen["recall_at_prompt"] == 0.5
    assert resumen["precision_respuesta"] == 0.5
    assert resumen["tasa_alucinacion"] == 1.0


def test_resumen_sin_casos_de_abstencion_no_divide_entre_cero():
    case = EvalCase("q1", "factual", "?", ["A.md"], ["x"])
    resumen = summarize([evaluate_case(case, "contiene x", ["A.md"], ["A.md"])])

    assert resumen["tasa_alucinacion"] is None


# --- carga del dataset -------------------------------------------------------

@pytest.mark.parametrize(
    "dataset", ["evals/faiss_docs/qa.es.jsonl", "evals/faiss_docs/qa.en.jsonl"]
)
def test_carga_los_datasets_reales(dataset):
    cases = load_cases(dataset)

    assert len(cases) == 30
    assert {c.type for c in cases} == {"factual", "multi_hop", "ausencia", "trampa"}

    for case in cases:
        if case.type in ("ausencia", "trampa"):
            assert case.must_cite == [], f"{case.id} no debe declarar must_cite"
        else:
            assert case.must_cite, f"{case.id} necesita must_cite"
            assert case.expected_any, f"{case.id} necesita expected_any"


def test_los_dos_datasets_son_equivalentes():
    """Ambos idiomas deben cubrir los mismos casos, para poder compararlos."""
    es = {c.id: c.type for c in load_cases("evals/faiss_docs/qa.es.jsonl")}
    en = {c.id: c.type for c in load_cases("evals/faiss_docs/qa.en.jsonl")}

    assert es == en


def test_must_cite_string_se_normaliza_a_lista():
    case = EvalCase.from_json(
        {"id": "x", "type": "factual", "question": "?", "must_cite": "A.md"}
    )
    assert case.must_cite == ["A.md"]


def test_dataset_invalido_falla_con_numero_de_linea(tmp_path):
    archivo = tmp_path / "malo.jsonl"
    archivo.write_text('{"id":"q1","type":"factual","question":"?"}\nno es json\n')

    with pytest.raises(ValueError, match="Línea 2"):
        load_cases(archivo)
