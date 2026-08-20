"""Puntuación de las respuestas del pipeline.

Las tres métricas se calculan por separado a propósito. Una respuesta puede
fallar porque el retriever no trajo el documento correcto o porque el modelo no
supo usarlo, y son problemas distintos con arreglos distintos. Mezclarlas en un
solo número esconde justamente lo que hay que diagnosticar.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from evals.dataset import EvalCase


# Frases con las que el modelo señala que no puede responder. La consigna del
# prompt le pide decir que "no tiene suficiente información", pero lo parafrasea
# de varias maneras, así que se reconocen las formas más habituales.
ABSTENTION_MARKERS = (
    "no_se",
    "no tengo suficiente",
    "no tengo la informacion",
    "no hay suficiente",
    "no hay informacion",
    # Con y sin el reflexivo: el modelo alterna entre "no se especifica" y
    # "el contexto no especifica". Reconocer solo una forma contaría
    # abstenciones legítimas como alucinaciones.
    "no especifica",
    "no menciona",
    "no indica",
    "no proporciona",
    "no detalla",
    "no esta en el contexto",
    "no aparece en el contexto",
    "no figura",
    "informacion insuficiente",
    "no puedo responder",
    "no es posible determinar",
)

# Bloque JSON dentro de la respuesta, con o sin cerca de markdown.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def normalize(text: str) -> str:
    """Minúsculas y sin acentos, para comparar sin depender de la tilde."""
    lowered = unicodedata.normalize("NFD", text.lower())

    return "".join(c for c in lowered if unicodedata.category(c) != "Mn")


def parse_answer(raw: str) -> tuple[str, list[str]]:
    """Extrae `answer` y `sources` del texto devuelto por el modelo.

    El pipeline pide un JSON pero recibe un string, y el modelo puede
    envolverlo en una cerca de markdown o acompañarlo de prosa. Si no se puede
    parsear, se devuelve el texto crudo: para puntuar una respuesta sirve igual,
    y marcar el caso como fallido escondería un acierto detrás de un problema
    de formato.
    """
    match = _JSON_BLOCK.search(raw)

    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            answer = payload.get("answer", "")
            sources = payload.get("sources", [])

            if not isinstance(sources, list):
                sources = [str(sources)]

            return str(answer) if answer else raw, [str(s) for s in sources]

    return raw, []


def is_abstention(answer: str) -> bool:
    """¿La respuesta dice que no hay información suficiente?"""
    normalized = normalize(answer)

    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def answer_is_correct(case: EvalCase, answer: str) -> bool:
    """¿La respuesta contiene el hecho esperado (o se abstiene, si tocaba)?"""
    if case.expects_abstention:
        return is_abstention(answer)

    # Abstenerse cuando sí había respuesta es un fallo, aunque alguna palabra
    # clave aparezca de pasada en la disculpa.
    if is_abstention(answer):
        return False

    normalized = normalize(answer)

    return any(normalize(option) in normalized for option in case.expected_any)


def citation_recall(case: EvalCase, docs: list[dict]) -> float | None:
    """Fracción de los archivos de `must_cite` presentes entre `docs`.

    Devuelve None cuando el caso no espera ninguna cita, para que esos casos
    no entren en el promedio en vez de contar como cero.
    """
    if not case.must_cite:
        return None

    sources = {doc.get("source", "") for doc in docs}

    encontrados = sum(1 for archivo in case.must_cite if archivo in sources)

    return encontrados / len(case.must_cite)


@dataclass
class CaseResult:
    """Resultado de evaluar un caso."""

    case: EvalCase
    answer: str
    sources_cited: list[str]

    # Recall sobre todo lo que devolvió el índice.
    recall_retrieved: float | None

    # Recall sobre los fragmentos que efectivamente entraron al prompt. Si este
    # es menor que el anterior, el rerank descartó documentos que sí servían.
    recall_context: float | None

    correct: bool
    abstained: bool
    prompt_tokens: int
    retrieved_sources: list[str] = field(default_factory=list)
    context_sources: list[str] = field(default_factory=list)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: list[CaseResult]) -> dict:
    """Agrega los resultados en las métricas del plan (§6)."""
    by_type: dict[str, dict] = {}

    for result in results:
        bucket = by_type.setdefault(
            result.case.type, {"total": 0, "correct": 0}
        )
        bucket["total"] += 1
        bucket["correct"] += int(result.correct)

    for bucket in by_type.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"]

    abstention_cases = [r for r in results if r.case.expects_abstention]

    # Alucinación: casos donde la respuesta correcta era abstenerse y el modelo
    # respondió igual.
    hallucinations = [r for r in abstention_cases if not r.abstained]

    return {
        "total": len(results),
        "accuracy": _mean([float(r.correct) for r in results]),
        "by_type": by_type,
        "recall_retrieved": _mean(
            [r.recall_retrieved for r in results if r.recall_retrieved is not None]
        ),
        "recall_context": _mean(
            [r.recall_context for r in results if r.recall_context is not None]
        ),
        "hallucination_rate": (
            len(hallucinations) / len(abstention_cases) if abstention_cases else None
        ),
        "hallucinated_ids": [r.case.id for r in hallucinations],
        "avg_prompt_tokens": _mean([float(r.prompt_tokens) for r in results]),
        # Los tokens de salida no están disponibles: `BaseLLM.generate` devuelve
        # un string y descarta el objeto de uso de la API. Se resuelve con el
        # rediseño de la interfaz (`docs/ARGUS-plan.md` §2.5).
        "avg_output_tokens": None,
    }
