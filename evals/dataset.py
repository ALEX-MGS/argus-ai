"""Carga del conjunto de evaluación en formato JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# Tipos cuya respuesta correcta es abstenerse.
ABSTENTION_TYPES = frozenset({"ausencia", "trampa"})


@dataclass
class EvalCase:
    """Un par pregunta/respuesta esperada del conjunto de evaluación."""

    id: str
    type: str
    question: str
    expected: str

    # Cualquiera de estas cadenas en la respuesta cuenta como acierto. Vacío
    # para los casos de abstención, que se puntúan de otra forma.
    expected_any: list[str] = field(default_factory=list)

    # Archivos del corpus que deberían sustentar la respuesta. Vacío cuando la
    # respuesta correcta es que no hay información.
    must_cite: list[str] = field(default_factory=list)

    @property
    def expects_abstention(self) -> bool:
        return self.type in ABSTENTION_TYPES


def _as_list(value) -> list[str]:
    """Normaliza `must_cite`, que puede venir como null, str o lista."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    return list(value)


def load_cases(path: str | Path) -> list[EvalCase]:
    """Lee el JSONL y devuelve los casos en el orden del archivo.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si alguna línea no es JSON válido o le falta un campo.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"No existe el conjunto de evaluación: {path}")

    cases: list[EvalCase] = []

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()

        if not line:
            continue

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} no es JSON válido: {error}") from error

        missing = {"id", "type", "q", "expected"} - raw.keys()

        if missing:
            raise ValueError(f"{path}:{number} le faltan campos: {sorted(missing)}")

        cases.append(
            EvalCase(
                id=raw["id"],
                type=raw["type"],
                question=raw["q"],
                expected=raw["expected"],
                expected_any=list(raw.get("expected_any", [])),
                must_cite=_as_list(raw.get("must_cite")),
            )
        )

    return cases
