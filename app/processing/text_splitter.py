"""Fragmentación de texto medida en tokens.

La versión anterior cortaba cada N caracteres por índice de string, lo que parte
palabras a la mitad y no guarda relación con el presupuesto de contexto del
modelo, que se cuenta en tokens.

Estrategia: se agrupan párrafos completos mientras quepan en el presupuesto.
Un párrafo que no cabe se divide por frases, y una frase que tampoco cabe se
corta por tokens como último recurso. El solapamiento se toma en unidades
completas (frases), no a media palabra.
"""

from __future__ import annotations

import re
from functools import lru_cache

import tiktoken


DEFAULT_ENCODING = "cl100k_base"

# Corta después de . ! ? : ; seguidos de espacio, y en saltos de línea.
_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@lru_cache(maxsize=4)
def _get_encoder(encoding_name: str):
    """Devuelve el codificador. Se cachea: construirlo es caro y es inmutable."""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Cuenta los tokens de un texto."""
    return len(_get_encoder(encoding_name).encode(text))


def _split_long_sentence(
    sentence: str, chunk_size: int, encoder
) -> list[str]:
    """Divide por tokens una frase que excede el presupuesto por sí sola.

    Es el único punto donde se puede partir una palabra. Solo ocurre con
    contenido sin puntuación, como tablas o bloques de código largos.
    """
    tokens = encoder.encode(sentence)
    pieces = []

    for start in range(0, len(tokens), chunk_size):
        piece = encoder.decode(tokens[start : start + chunk_size]).strip()
        if piece:
            pieces.append(piece)

    return pieces


def _size(units: list[tuple[str, int]], separator_tokens: int) -> int:
    """Tokens que ocupa un grupo de unidades ya unido por el separador."""
    if not units:
        return 0

    return sum(n for _, n in units) + separator_tokens * (len(units) - 1)


def _to_units(text: str, chunk_size: int, encoder) -> list[tuple[str, int]]:
    """Descompone el texto en unidades indivisibles con su conteo de tokens.

    Una unidad es un párrafo si cabe en el presupuesto; si no, una frase; y si
    una frase tampoco cabe, un corte duro por tokens.
    """
    units: list[tuple[str, int]] = []

    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        n_tokens = len(encoder.encode(paragraph))

        if n_tokens <= chunk_size:
            units.append((paragraph, n_tokens))
            continue

        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue

            n_sentence = len(encoder.encode(sentence))

            if n_sentence <= chunk_size:
                units.append((sentence, n_sentence))
            else:
                for piece in _split_long_sentence(sentence, chunk_size, encoder):
                    units.append((piece, len(encoder.encode(piece))))

    return units


def split_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[str]:
    """Divide un texto en fragmentos de a lo más `chunk_size` tokens.

    Args:
        text: texto a fragmentar.
        chunk_size: presupuesto máximo de tokens por fragmento.
        overlap: tokens de solapamiento entre fragmentos consecutivos, tomados
            en unidades completas. Debe ser menor que `chunk_size`.
        encoding_name: codificación de tiktoken a usar.

    Returns:
        Lista de fragmentos. Vacía si el texto no tiene contenido útil.

    Raises:
        ValueError: si `chunk_size` no es positivo o `overlap` no es menor.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que cero")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap debe estar entre 0 y chunk_size - 1")

    if not text or not text.strip():
        return []

    encoder = _get_encoder(encoding_name)
    units = _to_units(text, chunk_size, encoder)

    if not units:
        return []

    # El separador entre unidades también consume tokens: si no se contabiliza,
    # un fragmento con muchas unidades cortas excede el presupuesto.
    separator = "\n\n"
    separator_tokens = len(encoder.encode(separator))

    chunks: list[str] = []
    current: list[tuple[str, int]] = []

    for unit, n_tokens in units:
        entrante = (unit, n_tokens)

        if current and _size(current + [entrante], separator_tokens) > chunk_size:
            chunks.append(separator.join(u for u, _ in current))

            # Arrastra las últimas unidades hasta llenar el solapamiento.
            carried: list[tuple[str, int]] = []

            for previa in reversed(current):
                candidata = [previa] + carried

                if _size(candidata, separator_tokens) > overlap:
                    break

                carried = candidata

            # El arrastre no puede dejar sin espacio a la unidad entrante: se
            # recorta desde el inicio hasta que ambas quepan en el presupuesto.
            while carried and _size(carried + [entrante], separator_tokens) > chunk_size:
                carried.pop(0)

            current = carried

        current.append(entrante)

    if current:
        chunks.append("\n\n".join(u for u, _ in current))

    return chunks
