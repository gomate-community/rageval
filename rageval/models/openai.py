from __future__ import annotations

import asyncio
import pytest
import logging
import os
from abc import ABC
from dataclasses import dataclass, field

import openai
from langchain.schema import Generation, LLMResult

logger = logging.getLogger(__name__)


@dataclass
class OpenAILLM(ABC):
    """This is the OpenAI LLM model."""

    def __init__(self, model: str = "gpt-3.5-turbo-16k",
                 _api_key_env_var: str = field(default='NO_KEY', repr=False),
                 num_retries: int = 3,
                 timeout: int = 60) -> None:
        """Init the OpenAI Model."""
        self.model = model
        self._api_key_env_var = _api_key_env_var
        self.num_retries = num_retries
        self.timeout = timeout

        # api key
        self.api_key = os.getenv(self._api_key_env_var, 'NO_KEY')
        # key_from_env = os.getenv(self._api_key_env_var, 'NO_KEY')
        # self.api_key = key_from_env if key_from_env != 'NO_KEY' else self.api_key

    @property
    def llm(self):
        """Construct the OpenAI LLM model."""
        return openai.OpenAI(api_key=self.api_key)

    @pytest.mark.api
    def generate(self,
                 inputs: list(str),
                 system_role: str = "You are a helpful assistant") -> LLMResult:
        """Obtain the LLMResult from the response."""
        try:
            response = self.llm.with_options(
                max_retries=self.num_retries,
                timeout=self.timeout).chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": input_str} for input_str in inputs])
            result = self.create_llm_result(response)
            return result
        except openai.APIConnectionError as e:
            logger.info("The server could not be reached")
            logger.info(e.__cause__)  # an underlying Exception, likely raised within httpx.
            raise e
        except openai.RateLimitError as e:
            logger.info("A 429 status code was received; we should back off a bit.")
            raise e
        except openai.APIStatusError as e:
            logger.info("Another non-200-range status code was received")
            logger.info(e.status_code)
            logger.info(e.response)
            raise e

    def create_llm_result(self, response) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        if not isinstance(response, dict):
            response = response.model_dump()

        # token Usage
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model,
            "system_fingerprint": response.get("system_fingerprint", "")
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
        return LLMResult(generations=[generations], llm_output=llm_output)
