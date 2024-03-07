from __future__ import annotations

from typing import List, Optional, Any
import logging
import os
from abc import ABC
from dataclasses import dataclass, field

import openai
import pytest
from langchain.schema import Generation, LLMResult

logger = logging.getLogger(__name__)


@dataclass
class OpenAILLM(ABC):
    """This is the OpenAI LLM model."""

    def __init__(self, model: str = "gpt-3.5-turbo-16k",
                 _api_key_env_var: str = field(default='NO_KEY', repr=False),
                 num_retries: int = 3,
                 timeout: int = 60,
                 max_tokens: Optional[int] = None,
                 n:Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 logprobs: bool = False,
                 top_logprobs: Optional[int] = None) -> None:
        """Init the OpenAI Model."""
        self.model = model
        self.num_retries = num_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

        # api key
        self.api_key = os.getenv(_api_key_env_var, 'NO_KEY')

    @property
    def llm(self):
        """Construct the OpenAI LLM model."""
        return openai.OpenAI(api_key=self.api_key)

    @pytest.mark.api
    def generate(self,
                 inputs: List[str],
                 system_role: str = "You are a helpful assistant") -> LLMResult:
        """Obtain the LLMResult from the response."""
        messages = [{"role": "system", "content": system_role}, {"role": "user", "content": input_str} for input_str in inputs]
        try:
            response = self.llm.with_options(
                max_retries=self.num_retries,
                timeout=self.timeout).chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    n=self.n,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs)
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
