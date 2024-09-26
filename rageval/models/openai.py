from __future__ import annotations

from typing import List, Optional, Any, Union, Dict
import logging
import os
from abc import ABC
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import pytest
from tqdm import tqdm
from langchain.schema import Generation, LLMResult

logger = logging.getLogger(__name__)


@dataclass
class OpenAILLM(ABC):
    """This is the OpenAI LLM model. See more at https://platform.openai.com/docs/api-reference/.

    Args:
        model: str, The model name.
        _api_key_env_var: str, The environment variable that holds the api key.
        num_retries: int, The number of retries to make.
        timeout: int, The timeout for the request.

    Optional Args:
        max_tokens: int, The maximum number of tokens that can be generated in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
        n: int, How many chat completion choices to generate for each input message. Default to 1.
        temperature: float, What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.We generally recommend altering this or `top_p` but not both.
        top_p: float, An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with `top_p` probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. Default to 1.0.
        logprobs: bool or integer.
            For chat models, `logprobs` determines whether to return logprobs. Default to False. If true, returns the log probabilities of each output token returned in the content of message. This option is currently not available on the `gpt-4-vision-preview` model.
            For instruct models, `logprobs` is the number of the log probabilities of most likely output tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The maximum value for logprobs is 5.
        top_logprobs: int, only used in chat model. An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.
    """

    def __init__(self, model: str = "gpt-3.5-turbo",
                 _api_key_env_var: str = field(default='NO_KEY', repr=False),
                 base_url: str = "https://api.openai.com/v1",
                 num_retries: int = 3,
                 timeout: int = 60,
                 max_tokens: Optional[int] = None,
                 n: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 logprobs: Optional[Union[bool, int]] = None,
                 top_logprobs: Optional[int] = None) -> None:
        """Init the OpenAI Model."""
        self.model = model
        self.base_url = base_url
        self.num_retries = num_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        # api key
        self.api_key = os.getenv(_api_key_env_var, 'NO_KEY')

    @property
    def llm(self):
        """Construct the OpenAI LLM model."""
        return openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=self.num_retries,
            timeout=self.timeout
        )

    @property
    def _is_chat_model_engine(self) -> bool:
        if self.model == "gpt-3.5-turbo-instruct":
            return False
        elif self.model.startswith("gpt-3.5") or self.model.startswith("gpt-4"):
            return True
        return True

    def build_request(self, ) -> dict:
        """Build the request for the model."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": self.logprobs,
        }

    def _get_chat_model_response(self, prompt: List[Dict[str, str]]) -> dict:
        """Get the response from the chat model.

        The messages is a list of dictionaries, each containing a role and content key, for example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ].
        See more at https://platform.openai.com/docs/guides/chat-completions/getting-started.
        """
        request = self.build_request()
        request["messages"] = prompt
        request["top_logprobs"] = self.top_logprobs
        response = self.llm.chat.completions.create(**request)
        return response

    def _get_instruct_model_response(self, prompt: str) -> dict:
        request = self.build_request()
        request["prompt"] = prompt
        return self.llm.completions.create(**request)

    @pytest.mark.api
    def generate(self, prompt: Union[List[Dict[str, str]], str]) -> LLMResult:
        """
        Obtain the LLMResult from the response. If the response is not successful, return an empty generation.

        TODO: Add cache to the response.
        """
        result = LLMResult(generations=[[Generation(text="")]], llm_output={})
        try:
            if self._is_chat_model_engine:
                assert isinstance(prompt, list), "The prompts should be a list of dictionaries for chat model."
                response = self._get_chat_model_response(prompt=prompt)
            else:
                assert isinstance(prompt, str), "The prompts should be a string for instruct model."
                response = self._get_instruct_model_response(prompt=prompt)
            result = self.create_llm_result(response)
        except AssertionError as e:
            logger.info("Please check the input arguments.")
            logger.info(e)
        except openai.APIConnectionError as e:
            logger.info("The server could not be reached.")
            logger.info(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            logger.info("A 429 status code was received; we should back off a bit.")
            logger.info(e.__cause__)
        except openai.APIStatusError as e:
            logger.info("Another non-200-range status code was received.")
            logger.info(e.status_code)
            logger.info(e.response)
        except TypeError as e:
            logger.info("Please check the input arguments.")
            logger.info(e.__cause__)

        return result

    def create_llm_result(self, response) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        if not isinstance(response, dict):
            response = response.model_dump()

        # token Usage
        token_usage = response.get("usage", {})
        self.usage["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += token_usage.get("completion_tokens", 0)
        self.usage["total_tokens"] += token_usage.get("total_tokens", 0)
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model,
            "system_fingerprint": response.get("system_fingerprint", "")
        }

        choices = response["choices"]
        generations = [
            Generation(
                text=choice["message"]["content"] if self._is_chat_model_engine else choice["text"],
                generation_info=dict(
                    finish_reason=choice.get("finish_reason"),
                    logprobs=choice.get("logprobs"),
                ),
            )
            for choice in choices
        ]
        return LLMResult(generations=[generations], llm_output=llm_output)

    def calculate_api_cost(self):
        """
        Calculate the cost of the api usage.

        More detail for api prices: https://openai.com/pricing/
        """
        # $ / 1k tokens:
        mapping = {
            "gpt-3.5-turbo-0125": (0.0005, 0.0015),
            "gpt-4": (0.03, 0.06),
            "gpt-4-32k": (0.06, 0.12),
            "gpt-3.5-turbo-instruct": (0.0015, 0.002),
            "gpt-4o": (0.005, 0.015),
        }

        intokens = self.usage["prompt_tokens"]
        outtokens = self.usage["completion_tokens"]

        if self.model in mapping.keys():
            print(f"Total tokens: {self.usage['total_tokens']}")
            print(f"Input tokens: {intokens}, Output tokens: {outtokens}")
            print(f"Total cost: {mapping[self.model][0] * intokens / 1000 + mapping[self.model][1] * outtokens / 1000}")

    def batch_generate(self,
                       prompts: Union[List[List[Dict[str, str]]], List[str]],
                       max_workers: int = 1) -> List[LLMResult]:
        """Batch generate the LLMResult from the response using multithreading.

        Args:
            inputs: The list of inputs. If the model is chat model, the inputs should be a list of List[Dict[str, str]]. If the model is instruct model, the inputs should be a list of string.
            system_roles: List[str], The list of system roles. Default to None. If the model is chat model, the system_roles should be a list of strings. If the model is instruct model, the system_roles should be None.
            max_workers: int, The number of workers to use. Default to 1.

        Returns:
            List[LLMResult], The list of LLMResult. The order of the result is the same as the order of the inputs.

        """
        results = [None] * len(prompts)  # initialize the results
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, prompt in enumerate(prompts):
                futures.append((idx, executor.submit(self.generate, prompt)))

            for future in tqdm(as_completed([f[1] for f in futures]), total=len(prompts), desc="Generating"):
                idx = next(idx for idx, f in futures if f == future)
                result = future.result()
                results[idx] = result  # store the result at the correct order

        return results
