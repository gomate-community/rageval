from __future__ import annotations

import asyncio
import logging
import os
import typing as t
from abc import abstractmethod
from dataclasses import dataclass, field

import openai
from langchain.adapters.openai import convert_message_to_dict
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.schema import Generation, LLMResult

from ..llms import ragevalLLM


from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class OpenAIBase(ragevalLLM):
    def __init__(self, model: str, _api_key_env_var: str, timeout: int = 60) -> None:
        self.model = model
        self._api_key_env_var = _api_key_env_var
        self.timeout = timeout

        # api key
        key_from_env = os.getenv(self._api_key_env_var, 'NO_KEY')
        if key_from_env != 'NO_KEY':
            self.api_key = key_from_env
        else:
            self.api_key = self.api_key

    @property
    def llm(self):
        return self

    def create_llm_result(self, response) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        if not isinstance(response, dict):
            response = response.model_dump()

        # token Usage
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": None,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }

        choices = response["choices"]
        generations = [
            Generation(
                text=choice["message"]["content"],
                generation_info=dict(
                    finish_reason=choice.get("finish_reason"),
                    logprobs=choice.get("logprobs"),
                ),
            )
            for choice in choices
        ]
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return LLMResult(generations=[generations], llm_output=llm_output)

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 0,
    ) -> t.Any:  # TODO: LLMResult
        llm_results = [self.generate(p, n, temperature) for p in prompts]

        generations = [r.generations[0] for r in llm_results]
        return LLMResult(generations=generations)




@dataclass
class OpenAI(OpenAIBase):
    model: str = "gpt-3.5-turbo-16k"
    api_key: str = field(default='NO_KEY', repr=False)
    _api_key_env_var: str = "OPENAI_API_KEY"

    def validate_api_key(self):
        # before validating, check if the api key is already set
        api_key = os.getenv(self._api_key_env_var, 'NO_KEY')

