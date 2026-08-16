"""Carga de documentos desde el sistema de archivos.

Agnóstico al dominio: no conoce ningún corpus en particular. Recibe una ruta
y devuelve documentos con su origen, para que la capa de recuperación pueda
reportar de qué archivo salió cada fragmento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTENSIONS = (".md", ".txt")

# Directorios que nunca aportan contenido consultable.
DEFAULT_EXCLUDED_DIRS = (".git", "__pycache__", "node_modules", ".venv", "venv")


@dataclass
class Document:
    """Un archivo cargado, antes de fragmentarse."""

    text: str
    source: str  # ruta relativa a la raíz de carga, ej. "Faiss-indexes.md"


def load_documents(
    root: str | Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDED_DIRS,
    min_chars: int = 1,
) -> list[Document]:
    """Lee recursivamente los archivos de `root` y los devuelve como documentos.

    Args:
        root: carpeta raíz del corpus.
        extensions: extensiones a incluir (en minúsculas, con punto).
        exclude_dirs: nombres de carpeta a omitir en cualquier nivel.
        min_chars: descarta archivos con menos caracteres útiles que esto.

    Returns:
        Documentos ordenados por `source`, para que la indexación sea
        reproducible entre corridas.

    Raises:
        FileNotFoundError: si `root` no existe o no es una carpeta.
    """
    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de corpus: {root}")

    excluded = set(exclude_dirs)
    documents: list[Document] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        # Se omite si cualquier carpeta de la ruta está excluida.
        if excluded.intersection(path.relative_to(root).parts[:-1]):
            continue

        text = path.read_text(encoding="utf-8", errors="replace").strip()

        if len(text) < min_chars:
            continue

        documents.append(
            Document(text=text, source=str(path.relative_to(root)))
        )

    return documents
