"""Tests del cargador de documentos. No requieren llamadas a la API."""

import pytest

from app.ingestion.loader import load_documents


def _escribir(tmp_path, ruta_relativa: str, contenido: str):
    destino = tmp_path / ruta_relativa
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(contenido, encoding="utf-8")
    return destino


def test_carga_md_y_txt(tmp_path):
    _escribir(tmp_path, "uno.md", "contenido uno")
    _escribir(tmp_path, "dos.txt", "contenido dos")

    documentos = load_documents(tmp_path)

    assert {d.source for d in documentos} == {"uno.md", "dos.txt"}


def test_ignora_extensiones_no_incluidas(tmp_path):
    _escribir(tmp_path, "codigo.py", "print('hola')")
    _escribir(tmp_path, "doc.md", "contenido")

    documentos = load_documents(tmp_path)

    assert [d.source for d in documentos] == ["doc.md"]


def test_source_conserva_la_ruta_relativa(tmp_path):
    _escribir(tmp_path, "guias/indices.md", "contenido")

    documentos = load_documents(tmp_path)

    assert documentos[0].source == "guias/indices.md"


def test_excluye_directorios_indicados(tmp_path):
    _escribir(tmp_path, "bueno.md", "contenido útil")
    _escribir(tmp_path, "logs_bench_all_ivf/ruido.md", "volcado de benchmark")

    documentos = load_documents(tmp_path, exclude_dirs=("logs_bench_all_ivf",))

    assert [d.source for d in documentos] == ["bueno.md"]


def test_excluye_git_por_defecto(tmp_path):
    _escribir(tmp_path, "doc.md", "contenido")
    _escribir(tmp_path, ".git/interno.md", "no debe cargarse")

    documentos = load_documents(tmp_path)

    assert [d.source for d in documentos] == ["doc.md"]


def test_descarta_archivos_vacios(tmp_path):
    _escribir(tmp_path, "vacio.md", "   \n  ")
    _escribir(tmp_path, "lleno.md", "contenido")

    documentos = load_documents(tmp_path)

    assert [d.source for d in documentos] == ["lleno.md"]


def test_orden_reproducible(tmp_path):
    for nombre in ["c.md", "a.md", "b.md"]:
        _escribir(tmp_path, nombre, "contenido")

    documentos = load_documents(tmp_path)

    assert [d.source for d in documentos] == ["a.md", "b.md", "c.md"]


def test_carpeta_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "no_existe")
