# -*- coding: utf-8 -*-

import typing
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import floor

from datasets import Dataset
from tqdm import tqdm
from langchain.schema import LLMResult


@dataclass
class Metric(ABC):
    """Metric base class without LLM."""

    batch_size: int

    @abstractmethod
    def init_model(self, model: typing.Callable):
        """This method will lazy initialize the model."""
        ...

    def score(
        self,
        dataset: Dataset,
        batch_size: int = None,
    ) -> (float, Dataset):
        """Evaluate the dataset."""
        scores = []
        length = len(dataset)
        if batch_size:
            for start in tqdm(range(0, length, batch_size)):
                end = start + batch_size
                end = end if end < length else length
                score = self._score_batch(dataset.select(range(start, end)))
                scores.extend(score)
        else:
            scores = self._score_batch(dataset)

        return np.average(scores), dataset.add_column(f"{self.name}", scores)

    @abstractmethod
    def _score_batch(self, dataset: Dataset) -> list:
        ...


@dataclass
class MetricWithLLM(Metric):
    """Metrics based on LLM."""

    from rageval.models.openai import OpenAILLM

    # llm: ragevalLLM = field(default_factory=llm_factory)
    llm: OpenAILLM = OpenAILLM('gpt-3.5-turbo-16k', 'OPENAI_API_KEY')

    def init_model(self):
        """
        Initialize the LLM model.

        Init any models in the metric, this is invoked before evaluate()
        to load all the models
        Also check if the api key is valid for OpenAI and AzureOpenAI
        """
        if hasattr(self.llm, "validate_api_key"):
            self.llm.validate_api_key()
        if hasattr(self, "embeddings"):
            # since we are using Langchain Embeddings directly, we need to check this
            if hasattr(self.embeddings, "validate_api_key"):
                # TODO
                ...

    def parse_llm_result(self, prompts: [str], result: LLMResult):
        """Parse the LLM Result based on the Prompt."""
        ...
