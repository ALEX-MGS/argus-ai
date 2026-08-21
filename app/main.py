"""Chat por línea de comandos sobre el corpus indexado."""

from __future__ import annotations

import asyncio

from app.pipeline import DEFAULT_HISTORY_TURNS, build_default_pipeline


async def main() -> None:
    pipeline = build_default_pipeline()

    chat_history: list[str] = []

    while True:
        query = input("Haz una pregunta (o escribe salir): ")

        if query.lower() == "salir":
            break

        chat_history.append(f"Usuario: {query}")
        chat_history = chat_history[-DEFAULT_HISTORY_TURNS:]

        result = await pipeline.answer(query, chat_history)

        chat_history.append(f"Asistente: {result.answer}")
        chat_history = chat_history[-DEFAULT_HISTORY_TURNS:]

        print("\nRespuesta final:\n")
        print(result.answer)
        print(f"\nFuentes recuperadas: {', '.join(result.prompt_sources)}\n")


if __name__ == "__main__":
    asyncio.run(main())
