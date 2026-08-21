"""Arnés de evaluación.

Corre un conjunto de pares pregunta/respuesta contra el pipeline real y reporta
recuperación y generación por separado. Esa separación es el punto: sin ella no
se puede saber si un fallo vino del retriever o del modelo.

Uso:
    python -m evals.run_eval evals/faiss_docs/qa.jsonl
    python -m evals.run_eval <ruta> --retrieval-only   # sin llamar al LLM
    python -m evals.run_eval <ruta> --label "linea base"

Los resultados se guardan en evals/results/ con el commit de git, para que cada
corrida sea comparable contra las anteriores.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.pipeline import RagPipeline, build_default_pipeline


RESULTS_DIR = Path("evals/results")

# Marcadores de abstención, en los dos idiomas en que puede responder el modelo.
ABSTENTION_MARKERS = (
    "no tengo suficiente informaci",
    "no hay suficiente informaci",
    "no se proporciona",
    "no se especifica",
    "no está en el contexto",
    "no esta en el contexto",
    "not enough information",
    "insufficient information",
    "does not provide",
    "not specified",
    "no information",
)

ABSTENTION_TYPES = ("ausencia", "trampa")


@dataclass
class EvalCase:
    """Un par de evaluación leído del JSONL."""

    id: str
    type: str
    question: str
    must_cite: list[str]
    expected_any: list[str]
    note: str = ""

    @classmethod
    def from_json(cls, raw: dict) -> EvalCase:
        must_cite = raw.get("must_cite")

        if must_cite is None:
            must_cite = []
        elif isinstance(must_cite, str):
            must_cite = [must_cite]

        return cls(
            id=raw["id"],
            type=raw["type"],
            question=raw["question"],
            must_cite=must_cite,
            expected_any=raw.get("expected_any", []),
            note=raw.get("note", ""),
        )


@dataclass
class CaseOutcome:
    """Lo medido para un caso."""

    id: str
    type: str
    question: str
    answer: str = ""
    prompt_sources: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    must_cite: list[str] = field(default_factory=list)
    recall_at_prompt: bool | None = None
    recall_at_k: bool | None = None
    answer_correct: bool | None = None
    abstained: bool | None = None
    error: str = ""


def load_cases(path: str | Path) -> list[EvalCase]:
    """Lee el JSONL de pares de evaluación."""
    cases = []

    for numero, linea in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        linea = linea.strip()

        if not linea or linea.startswith("//"):
            continue

        try:
            cases.append(EvalCase.from_json(json.loads(linea)))
        except (json.JSONDecodeError, KeyError) as error:
            raise ValueError(f"Línea {numero} inválida en {path}: {error}") from error

    return cases


def extract_answer_text(raw_response: str) -> str:
    """Saca el campo `answer` del JSON que devuelve el modelo.

    El prompt pide JSON, pero nada lo garantiza. Si no se puede parsear se
    devuelve la respuesta cruda: para calificar por substring da igual, y así
    un fallo de formato no se confunde con un fallo de contenido.
    """
    texto = raw_response.strip()

    if texto.startswith("```"):
        texto = texto.split("```")[1] if "```" in texto[3:] else texto.strip("`")
        texto = texto.removeprefix("json").strip()

    try:
        datos = json.loads(texto)
    except json.JSONDecodeError:
        return raw_response

    if isinstance(datos, dict) and "answer" in datos:
        return str(datos["answer"])

    return raw_response


def is_abstention(answer: str) -> bool:
    """¿La respuesta admite no tener información?"""
    minuscula = answer.lower()
    return any(marker in minuscula for marker in ABSTENTION_MARKERS)


def matches_expected(answer: str, expected_any: list[str]) -> bool:
    """Calificación por substring, insensible a mayúsculas."""
    minuscula = answer.lower()
    return any(esperado.lower() in minuscula for esperado in expected_any)


def evaluate_case(case: EvalCase, answer_text: str,
                  prompt_sources: list[str], retrieved_sources: list[str]) -> CaseOutcome:
    """Calcula las métricas de un caso ya ejecutado."""
    outcome = CaseOutcome(
        id=case.id,
        type=case.type,
        question=case.question,
        answer=answer_text,
        prompt_sources=prompt_sources,
        retrieved_sources=retrieved_sources,
        must_cite=case.must_cite,
    )

    if case.must_cite:
        # Basta con que uno de los documentos esperados haya sido recuperado.
        outcome.recall_at_prompt = any(d in prompt_sources for d in case.must_cite)
        outcome.recall_at_k = any(d in retrieved_sources for d in case.must_cite)

    outcome.abstained = is_abstention(answer_text)

    if case.type in ABSTENTION_TYPES:
        # Aquí lo correcto es admitir que no se sabe.
        outcome.answer_correct = outcome.abstained
    else:
        outcome.answer_correct = matches_expected(answer_text, case.expected_any)

    return outcome


async def run_cases(cases: list[EvalCase], pipeline: RagPipeline,
                    retrieval_only: bool) -> list[CaseOutcome]:
    """Ejecuta todos los casos contra el pipeline."""
    outcomes = []

    for case in cases:
        print(f"  {case.id} ... ", end="", flush=True)

        try:
            if retrieval_only:
                vector = await pipeline.embedding_service.embed(case.question)
                docs = pipeline.vector_store.search(
                    vector, k=pipeline.k, threshold=pipeline.threshold
                )
                from app.pipeline import rerank

                reranked = rerank(case.question, docs)
                retrieved = [d.get("source", "unknown") for d in reranked]
                prompt_sources = retrieved[: pipeline.top_docs]

                outcome = CaseOutcome(
                    id=case.id,
                    type=case.type,
                    question=case.question,
                    prompt_sources=prompt_sources,
                    retrieved_sources=retrieved,
                    must_cite=case.must_cite,
                )

                if case.must_cite:
                    outcome.recall_at_prompt = any(d in prompt_sources for d in case.must_cite)
                    outcome.recall_at_k = any(d in retrieved for d in case.must_cite)
            else:
                result = await pipeline.answer(case.question)
                answer_text = extract_answer_text(result.answer)

                outcome = evaluate_case(
                    case, answer_text, result.prompt_sources, result.retrieved_sources
                )

            print("ok")

        except Exception as error:  # noqa: BLE001 - un caso roto no debe tumbar la corrida
            outcome = CaseOutcome(
                id=case.id, type=case.type, question=case.question, error=str(error)
            )
            print(f"ERROR: {error}")

        outcomes.append(outcome)

    return outcomes


def summarize(outcomes: list[CaseOutcome]) -> dict:
    """Agrega las métricas de la corrida."""
    con_must_cite = [o for o in outcomes if o.must_cite and not o.error]
    respondibles = [
        o for o in outcomes if o.type not in ABSTENTION_TYPES and not o.error
    ]
    abstencion = [o for o in outcomes if o.type in ABSTENTION_TYPES and not o.error]

    def ratio(numerador: int, denominador: int) -> float | None:
        return round(numerador / denominador, 3) if denominador else None

    return {
        "casos": len(outcomes),
        "errores": sum(1 for o in outcomes if o.error),
        "recall_at_prompt": ratio(
            sum(1 for o in con_must_cite if o.recall_at_prompt), len(con_must_cite)
        ),
        "recall_at_k": ratio(
            sum(1 for o in con_must_cite if o.recall_at_k), len(con_must_cite)
        ),
        "precision_respuesta": ratio(
            sum(1 for o in respondibles if o.answer_correct), len(respondibles)
        ),
        "tasa_alucinacion": ratio(
            sum(1 for o in abstencion if not o.answer_correct), len(abstencion)
        ),
    }


def git_commit() -> str:
    """Commit actual, para poder atribuir cada corrida a un estado del código."""
    try:
        salida = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        sucio = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        marca = "-sucio" if sucio.stdout.strip() else ""
        return salida.stdout.strip() + marca
    except Exception:  # noqa: BLE001 - fuera de un repo o sin git
        return "desconocido"


def print_report(outcomes: list[CaseOutcome], resumen: dict) -> None:
    """Imprime el detalle por caso y el resumen."""
    print("\n" + "=" * 78)
    print(f"{'id':5} {'tipo':12} {'recall@3':9} {'recall@10':10} {'respuesta':10}")
    print("-" * 78)

    def marca(valor: bool | None) -> str:
        if valor is None:
            return "  -"
        return "  ok" if valor else "  NO"

    for o in outcomes:
        if o.error:
            print(f"{o.id:5} {o.type:12} ERROR: {o.error[:40]}")
            continue

        print(
            f"{o.id:5} {o.type:12} {marca(o.recall_at_prompt):9} "
            f"{marca(o.recall_at_k):10} {marca(o.answer_correct):10}"
        )

    print("=" * 78)
    print(f"Recall@3 (lo que llegó al prompt) : {resumen['recall_at_prompt']}")
    print(f"Recall@10 (lo recuperado)         : {resumen['recall_at_k']}")
    print(f"Precisión de respuesta            : {resumen['precision_respuesta']}")
    print(f"Tasa de alucinación               : {resumen['tasa_alucinacion']}")

    if resumen["errores"]:
        print(f"Casos con error                   : {resumen['errores']}")


def save_results(outcomes: list[CaseOutcome], resumen: dict,
                 dataset: str, label: str, retrieval_only: bool) -> Path:
    """Guarda la corrida en disco, identificada por commit y fecha."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    momento = datetime.now(timezone.utc)
    commit = git_commit()
    destino = RESULTS_DIR / f"{momento:%Y%m%d-%H%M%S}_{commit}.json"

    destino.write_text(
        json.dumps(
            {
                "fecha_utc": momento.isoformat(),
                "commit": commit,
                "dataset": dataset,
                "etiqueta": label,
                "modo": "solo_recuperacion" if retrieval_only else "completo",
                "resumen": resumen,
                "casos": [asdict(o) for o in outcomes],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return destino


async def main() -> None:
    parser = argparse.ArgumentParser(description="Corre el conjunto de evaluación.")
    parser.add_argument("dataset", help="Ruta al JSONL de pares")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Solo mide recuperación; no llama al LLM")
    parser.add_argument("--label", default="", help="Etiqueta para identificar la corrida")

    args = parser.parse_args()

    cases = load_cases(args.dataset)
    print(f"Casos cargados: {len(cases)}\n")

    pipeline = build_default_pipeline()

    outcomes = await run_cases(cases, pipeline, args.retrieval_only)
    resumen = summarize(outcomes)

    print_report(outcomes, resumen)

    destino = save_results(
        outcomes, resumen, args.dataset, args.label, args.retrieval_only
    )
    print(f"\nResultados guardados en {destino}")


if __name__ == "__main__":
    asyncio.run(main())
