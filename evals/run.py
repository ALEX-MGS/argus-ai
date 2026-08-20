"""Corre el conjunto de evaluación contra el pipeline y reporta las métricas.

Uso:

    # Verificación de la tubería, sin API ni costo:
    python -m evals.run --stub

    # Corrida real (requiere OPENAI_API_KEY y un índice ya construido):
    python -m app.embeddings.index_documents evals/faiss_docs/corpus \\
        --exclude logs_bench_all_ivf
    python -m evals.run

El reporte se guarda en `evals/results/` para poder comparar corridas.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.pipeline import (
    DEFAULT_K,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_N,
    RAGPipeline,
)
from app.processing.text_splitter import count_tokens
from evals.dataset import EvalCase, load_cases
from evals.scoring import (
    CaseResult,
    answer_is_correct,
    citation_recall,
    is_abstention,
    parse_answer,
    summarize,
)


DEFAULT_DATASET = "evals/faiss_docs/qa.jsonl"
DEFAULT_CORPUS = "evals/faiss_docs/corpus"
DEFAULT_RESULTS_DIR = "evals/results"

# El wiki de FAISS incluye volcados de benchmark que no aportan contenido
# consultable y sí distorsionan la recuperación (`docs/ARGUS-plan.md` §5).
DEFAULT_EXCLUDE = ("logs_bench_all_ivf",)


def build_stub_store(corpus: str, exclude: tuple[str, ...], chunk_size: int,
                     overlap: int):
    """Indexa el corpus en memoria con embeddings falsos y deterministas."""
    from app.embeddings.index_documents import build_chunks
    from app.embeddings.vector_store import VectorStore
    from app.ingestion.loader import DEFAULT_EXCLUDED_DIRS
    from evals.stubs import DEFAULT_DIMENSION, hashed_embedding

    chunks = build_chunks(
        corpus, chunk_size, overlap, DEFAULT_EXCLUDED_DIRS + exclude
    )

    store = VectorStore(dimension=DEFAULT_DIMENSION)

    for text, source in chunks:
        store.add(hashed_embedding(text, DEFAULT_DIMENSION), text, source=source)

    return store


async def evaluate_case(pipeline: RAGPipeline, case: EvalCase) -> CaseResult:
    """Corre un caso y lo puntúa. Sin historial: la evaluación es de un turno."""
    result = await pipeline.answer(case.question, history=[])

    answer, sources_cited = parse_answer(result.answer)

    return CaseResult(
        case=case,
        answer=answer,
        sources_cited=sources_cited,
        recall_retrieved=citation_recall(case, result.retrieved),
        recall_context=citation_recall(case, result.context),
        correct=answer_is_correct(case, answer),
        abstained=is_abstention(answer),
        prompt_tokens=count_tokens(result.prompt),
        retrieved_sources=[d.get("source", "") for d in result.retrieved],
        context_sources=[d.get("source", "") for d in result.context],
    )


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/d" if value is None else f"{value:.2f}{suffix}"


def print_report(results: list[CaseResult], summary: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'id':<6} {'tipo':<10} {'ok':<4} {'rec@k':<7} {'rec@ctx':<8} respuesta")
    print("-" * 78)

    for r in results:
        marca = "SI" if r.correct else "NO"
        recorte = r.answer.replace("\n", " ")[:30]
        print(
            f"{r.case.id:<6} {r.case.type:<10} {marca:<4} "
            f"{_fmt(r.recall_retrieved):<7} {_fmt(r.recall_context):<8} {recorte}"
        )

    print("=" * 78)
    print(f"Casos:                     {summary['total']}")
    print(f"Precisión global:          {_fmt(summary['accuracy'])}")

    for tipo, datos in sorted(summary["by_type"].items()):
        print(
            f"  {tipo:<12} {datos['correct']}/{datos['total']} "
            f"({datos['accuracy']:.0%})"
        )

    print(f"Recall@k (recuperados):    {_fmt(summary['recall_retrieved'])}")
    print(f"Recall@n (en el prompt):   {_fmt(summary['recall_context'])}")
    print(f"Tasa de alucinación:       {_fmt(summary['hallucination_rate'])}")

    if summary["hallucinated_ids"]:
        print(f"  alucinados:              {', '.join(summary['hallucinated_ids'])}")

    print(f"Tokens de prompt (media):  {_fmt(summary['avg_prompt_tokens'])}")
    print(f"Tokens de salida (media):  n/d (BaseLLM descarta el uso de la API)")

    caida = (
        summary["recall_retrieved"] is not None
        and summary["recall_context"] is not None
        and summary["recall_retrieved"] > summary["recall_context"]
    )

    if caida:
        print(
            "\nNota: el recall baja entre lo recuperado y lo que llega al prompt.\n"
            "      Esa diferencia es responsabilidad del rerank y del corte top-n."
        )


def write_report(path: Path, summary: dict, results: list[CaseResult],
                 config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": summary,
        "cases": [
            {
                "id": r.case.id,
                "type": r.case.type,
                "question": r.case.question,
                "expected": r.case.expected,
                "must_cite": r.case.must_cite,
                "answer": r.answer,
                "sources_cited": r.sources_cited,
                "retrieved_sources": r.retrieved_sources,
                "context_sources": r.context_sources,
                "recall_retrieved": r.recall_retrieved,
                "recall_context": r.recall_context,
                "correct": r.correct,
                "abstained": r.abstained,
                "prompt_tokens": r.prompt_tokens,
            }
            for r in results
        ],
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nReporte guardado en {path}")


def parse_threshold(value: str) -> float | None:
    return None if value.lower() in {"none", "null", "ninguno"} else float(value)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el pipeline RAG.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--stub", action="store_true",
                        help="Usa dobles deterministas: sin API y sin costo")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--threshold", type=parse_threshold,
                        default=DEFAULT_THRESHOLD,
                        help="Umbral de distancia, o 'none' para desactivarlo")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None,
                        help="Corre solo los primeros N casos")
    parser.add_argument("--out", default=None, help="Ruta del reporte JSON")

    args = parser.parse_args()

    cases = load_cases(args.dataset)

    if args.limit:
        cases = cases[: args.limit]

    if args.stub:
        from evals.stubs import StubEmbeddingService, StubLLM

        store = build_stub_store(
            args.corpus, DEFAULT_EXCLUDE, args.chunk_size, args.overlap
        )
        embedding_service = StubEmbeddingService()
        llm = StubLLM()
    else:
        from app.embeddings.embedding_service import EmbeddingService
        from app.embeddings.vector_store import VectorStore
        from app.models.openai_llm import OpenAILLM

        store = VectorStore()
        store.load()
        embedding_service = EmbeddingService()
        llm = OpenAILLM()

    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        vector_store=store,
        llm=llm,
        k=args.k,
        threshold=args.threshold,
        top_n=args.top_n,
    )

    print(f"Evaluando {len(cases)} casos "
          f"({'dobles' if args.stub else 'API real'})...")

    # En serie a propósito: en la corrida real, lanzar 30 consultas en paralelo
    # invita a que la API responda con rate limit, y el pipeline todavía no
    # tiene reintentos.
    results = [await evaluate_case(pipeline, case) for case in cases]

    summary = summarize(results)

    print_report(results, summary)

    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sufijo = "stub" if args.stub else "real"
    destino = Path(args.out or f"{DEFAULT_RESULTS_DIR}/{marca}-{sufijo}.json")

    write_report(
        destino,
        summary,
        results,
        config={
            "dataset": args.dataset,
            "corpus": args.corpus,
            "stub": args.stub,
            "k": args.k,
            "top_n": args.top_n,
            "threshold": args.threshold,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
