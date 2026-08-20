"""Tests del conjunto de evaluación: carga y coherencia con el corpus."""

from pathlib import Path

import pytest

from evals.dataset import load_cases


RAIZ = Path(__file__).resolve().parent.parent
QA = RAIZ / "evals" / "faiss_docs" / "qa.jsonl"
CORPUS = RAIZ / "evals" / "faiss_docs" / "corpus"


# --- carga -----------------------------------------------------------------


def test_must_cite_nulo_queda_como_lista_vacia(tmp_path):
    archivo = tmp_path / "qa.jsonl"
    archivo.write_text(
        '{"id":"a","type":"ausencia","q":"¿?","expected":"NO_SE","must_cite":null}\n',
        encoding="utf-8",
    )

    caso = load_cases(archivo)[0]

    assert caso.must_cite == []
    assert caso.expects_abstention


def test_must_cite_string_queda_como_lista(tmp_path):
    archivo = tmp_path / "qa.jsonl"
    archivo.write_text(
        '{"id":"f","type":"factual","q":"¿?","expected":"x","must_cite":"a.md"}\n',
        encoding="utf-8",
    )

    assert load_cases(archivo)[0].must_cite == ["a.md"]


def test_ignora_lineas_vacias(tmp_path):
    archivo = tmp_path / "qa.jsonl"
    archivo.write_text(
        '\n{"id":"f","type":"factual","q":"¿?","expected":"x"}\n\n',
        encoding="utf-8",
    )

    assert len(load_cases(archivo)) == 1


def test_json_invalido_señala_la_linea(tmp_path):
    archivo = tmp_path / "qa.jsonl"
    archivo.write_text("{no es json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=":1"):
        load_cases(archivo)


def test_campo_faltante_se_reporta(tmp_path):
    archivo = tmp_path / "qa.jsonl"
    archivo.write_text('{"id":"f","type":"factual"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected"):
        load_cases(archivo)


def test_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "no_existe.jsonl")


# --- coherencia del conjunto real -----------------------------------------


def test_el_conjunto_real_carga():
    casos = load_cases(QA)

    assert len(casos) >= 30
    assert len({c.id for c in casos}) == len(casos), "hay ids repetidos"


def test_distribucion_de_tipos():
    """El sesgo hacia casos difíciles es intencional (`ARGUS-plan.md` §6)."""
    casos = load_cases(QA)

    conteo = {}
    for caso in casos:
        conteo[caso.type] = conteo.get(caso.type, 0) + 1

    assert conteo == {"factual": 15, "multi_hop": 8, "ausencia": 5, "trampa": 2}


def test_los_casos_con_respuesta_traen_criterio_de_acierto():
    for caso in load_cases(QA):
        if caso.expects_abstention:
            assert not caso.expected_any, f"{caso.id}: abstención no lleva keywords"
            assert not caso.must_cite, f"{caso.id}: abstención no debe citar"
        else:
            assert caso.expected_any, f"{caso.id}: sin expected_any no se puede puntuar"
            assert caso.must_cite, f"{caso.id}: sin must_cite no se puede medir recall"


def test_multi_hop_exige_al_menos_dos_fuentes():
    for caso in load_cases(QA):
        if caso.type == "multi_hop":
            assert len(caso.must_cite) >= 2, f"{caso.id} no es multi-hop de verdad"


@pytest.mark.skipif(not CORPUS.is_dir(), reason="el corpus no está clonado")
def test_todo_must_cite_existe_en_el_corpus():
    """Un nombre de archivo mal escrito daría recall 0 sin ningún error visible.

    Es el fallo más silencioso posible del conjunto de evaluación: las métricas
    saldrían pésimas y la culpa parecería del retriever.
    """
    disponibles = {
        str(p.relative_to(CORPUS)) for p in CORPUS.rglob("*.md")
    }

    faltantes = {
        (caso.id, archivo)
        for caso in load_cases(QA)
        for archivo in caso.must_cite
        if archivo not in disponibles
    }

    assert not faltantes, f"must_cite que no existen en el corpus: {sorted(faltantes)}"
