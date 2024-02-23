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

    @property
    @abstractmethod
    def name(self) -> str:
        """The metric name."""
        ...

    def compute(
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
                score = self._compute_batch(dataset.select(range(start, end)))
                scores.extend(score)
        else:
            scores = self._compute_batch(dataset)

        return np.average(scores), dataset.add_column(f"{self.name}", scores)

    @abstractmethod
    def _compute_batch(self, dataset: Dataset) -> list:
        ...


@dataclass
class MetricWithLLM(Metric):
    """Metrics based on LLM."""

    @abstractmethod
    def init_model(self, model: typing.Callable):
        """This method will lazy initialize the model."""
        ...

    def parse_llm_result(self, prompts: [str], result: LLMResult):
        """Parse the LLM Result based on the Prompt."""
        ...
