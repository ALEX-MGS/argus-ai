"""Construcción del índice vectorial a partir de una carpeta de documentos.

Uso:
    python -m app.embeddings.index_documents evals/faiss_docs/corpus
    python -m app.embeddings.index_documents <ruta> --exclude logs_bench_all_ivf
    python -m app.embeddings.index_documents <ruta> --dry-run

`--dry-run` recorre y fragmenta el corpus sin llamar a la API, para revisar
cuántos fragmentos saldrían y de qué tamaño antes de gastar en embeddings.
"""

from __future__ import annotations

import argparse
import asyncio

from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.vector_store import VectorStore
from app.ingestion.loader import DEFAULT_EXCLUDED_DIRS, load_documents
from app.processing.text_splitter import count_tokens, split_text


def build_chunks(
    root: str,
    chunk_size: int,
    overlap: int,
    exclude_dirs: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Carga y fragmenta el corpus. Devuelve pares (texto, archivo de origen)."""
    documents = load_documents(root, exclude_dirs=exclude_dirs)

    chunks: list[tuple[str, str]] = []

    for document in documents:
        for chunk in split_text(document.text, chunk_size, overlap):
            chunks.append((chunk, document.source))

    print(f"Documentos cargados: {len(documents)}")
    print(f"Fragmentos generados: {len(chunks)}")

    return chunks


def report_chunks(chunks: list[tuple[str, str]]) -> None:
    """Imprime estadísticas de fragmentación sin llamar a la API."""
    if not chunks:
        print("No se generó ningún fragmento.")
        return

    sizes = [count_tokens(text) for text, _ in chunks]

    print(f"Tokens por fragmento — mín: {min(sizes)} | "
          f"máx: {max(sizes)} | promedio: {sum(sizes) // len(sizes)}")
    print(f"Tokens totales a embeber: {sum(sizes)}")

    print("\nMuestra del primer fragmento:")
    print(f"[{chunks[0][1]}] {chunks[0][0][:300]}...")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indexa una carpeta de documentos en el índice vectorial."
    )
    parser.add_argument("path", help="Carpeta raíz del corpus")
    parser.add_argument("--chunk-size", type=int, default=400,
                        help="Tokens máximos por fragmento (default: 400)")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Tokens de solapamiento (default: 50)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Carpetas adicionales a omitir")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fragmenta y reporta sin llamar a la API")

    args = parser.parse_args()

    exclude_dirs = DEFAULT_EXCLUDED_DIRS + tuple(args.exclude)

    chunks = build_chunks(args.path, args.chunk_size, args.overlap, exclude_dirs)

    if not chunks:
        print("Nada que indexar.")
        return

    if args.dry_run:
        report_chunks(chunks)
        print("\nDry run: no se llamó a la API ni se guardó índice.")
        return

    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    vectors = await embedding_service.embed_batch([text for text, _ in chunks])

    for vector, (text, source) in zip(vectors, chunks):
        vector_store.add(vector, text, source=source)

    vector_store.save()

    print(f"Índice creado y guardado con {len(chunks)} fragmentos")


if __name__ == "__main__":
    asyncio.run(main())
