from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from personaagent.config import Config

def create_llm() -> BaseChatModel:
    return ChatOllama(
        model=Config.MODEL_NAME,
        temperature=Config.TEMPERATURE,
        num_ctx=Config.CONTEXT_WINDOW,
        verbose=False,
        keep_alive=-1,
        streaming=True,
    )