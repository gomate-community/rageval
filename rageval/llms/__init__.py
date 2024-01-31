from rageval.llms.base import ragevalLLM
from rageval.llms.openai import OpenAI

__all__ = ["ragevalLLM", "OpenAI"]


def llm_factory(model="gpt-3.5-turbo-16k") -> ragevalLLM:
    return OpenAI(model=model)
