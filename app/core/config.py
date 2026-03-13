from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    API_KEY = os.getenv("OPENAI_API_KEY")

settings = Settings()
