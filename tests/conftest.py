"""Configuración común de tests.

Asegura que el paquete `app` sea importable al correr pytest desde cualquier
directorio del proyecto.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
