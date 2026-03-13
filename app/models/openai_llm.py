from openai import AsyncOpenAI
from app.models.base_llm import BaseLLM
from app.core.config import settings


class OpenAILLM(BaseLLM):

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.API_KEY)

    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content
