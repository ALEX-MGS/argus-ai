"""Tests del fragmentador. No requieren llamadas a la API de OpenAI.

tiktoken sí descarga su tabla de codificación la primera vez (y la deja en
caché). Sin red y sin caché previa, la suite se omite en lugar de fallar.
"""

import pytest
import tiktoken

from app.processing.text_splitter import count_tokens, split_text

try:
    tiktoken.get_encoding("cl100k_base")
except Exception as error:  # noqa: BLE001 - red, permisos o datos corruptos
    pytest.skip(
        f"tiktoken no pudo cargar la codificación: {error}",
        allow_module_level=True,
    )


def test_texto_vacio_no_produce_fragmentos():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_texto_corto_queda_en_un_solo_fragmento():
    text = "FAISS es una biblioteca para búsqueda eficiente de vectores."
    assert split_text(text, chunk_size=400) == [text]


def test_ningun_fragmento_excede_el_presupuesto():
    text = "\n\n".join(
        f"Párrafo número {i} con contenido suficiente para ocupar tokens." * 5
        for i in range(40)
    )

    chunks = split_text(text, chunk_size=200, overlap=20)

    assert chunks
    assert all(count_tokens(chunk) <= 200 for chunk in chunks)


def test_unidad_grande_despues_de_corte_no_excede_el_presupuesto():
    """El solapamiento arrastrado no debe robarle espacio a la unidad entrante.

    Si tras cerrar un fragmento se arrastran `overlap` tokens y la siguiente
    unidad ocupa casi todo el presupuesto, el fragmento resultante se pasaba
    por el tamaño del arrastre.
    """
    grande = " ".join(f"palabra{i}" for i in range(340)) + "."
    chico = "Nota breve."

    text = "\n\n".join([chico, grande, chico, grande, chico, grande])

    chunks = split_text(text, chunk_size=400, overlap=50)

    excedidos = [count_tokens(c) for c in chunks if count_tokens(c) > 400]

    assert not excedidos, f"fragmentos de {excedidos} tokens con chunk_size=400"


def test_muchas_unidades_cortas_no_exceden_el_presupuesto():
    """Los separadores entre unidades también consumen presupuesto.

    Con párrafos muy cortos hay muchos separadores por fragmento; si no se
    contabilizan, el fragmento resultante supera el chunk_size declarado.
    """
    text = "\n\n".join(f"Línea {i}." for i in range(400))

    for chunk_size in (50, 120, 400):
        chunks = split_text(text, chunk_size=chunk_size, overlap=10)

        excedidos = [count_tokens(c) for c in chunks if count_tokens(c) > chunk_size]

        assert not excedidos, (
            f"con chunk_size={chunk_size} hubo fragmentos de {excedidos} tokens"
        )


def test_no_se_parten_palabras_en_texto_con_puntuacion():
    """El corte duro por tokens solo debe ocurrir sin puntuación."""
    text = " ".join(f"palabra{i} es un término completo." for i in range(200))

    chunks = split_text(text, chunk_size=100, overlap=10)
    reconstruido = " ".join(chunks)

    # Toda palabra original sigue apareciendo entera en algún fragmento.
    for i in range(200):
        assert f"palabra{i}" in reconstruido


def test_hay_solapamiento_entre_fragmentos_consecutivos():
    text = "\n\n".join(f"Este es el párrafo distinto número {i}." for i in range(60))

    chunks = split_text(text, chunk_size=100, overlap=40)

    assert len(chunks) > 1

    # El inicio de un fragmento debe reaparecer al final del anterior.
    solapados = sum(
        1
        for anterior, siguiente in zip(chunks, chunks[1:])
        if siguiente.split("\n\n")[0] in anterior
    )

    assert solapados > 0


def test_cubre_todo_el_contenido():
    text = "\n\n".join(f"Contenido único {i} que no debe perderse." for i in range(50))

    chunks = split_text(text, chunk_size=120, overlap=20)

    for i in range(50):
        assert any(f"Contenido único {i} " in chunk for chunk in chunks)


def test_frase_sin_puntuacion_mas_larga_que_el_presupuesto_se_corta():
    text = "token " * 500  # sin puntuación: obliga al corte duro

    chunks = split_text(text, chunk_size=50, overlap=0)

    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 50 for chunk in chunks)


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        split_text("texto", chunk_size=0)

    with pytest.raises(ValueError):
        split_text("texto", chunk_size=100, overlap=100)

    with pytest.raises(ValueError):
        split_text("texto", chunk_size=100, overlap=-1)
