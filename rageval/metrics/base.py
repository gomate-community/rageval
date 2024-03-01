# -*- coding: utf-8 -*-

from typing import List, Tuple, Callable, Optional
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from datasets import Dataset, MetricInfo
from datasets.metric import MetricInfoMixin
from datasets.naming import camelcase_to_snakecase
from langchain.schema import LLMResult
from tqdm import tqdm


def add_attribute(attribute_name, attribute_value):
    """
    This decorate is used to set attribute for Class.

    Currently, this decorate can be used to set attr:metric_type for each metric.
    There are four types, i.e., 'AnswerCorrectness', 'AnswerGroundedness', 'ContextRelevancy', 'ContextAdequacy', for all RAG metrics.
    """
    def decorator(cls):
        setattr(cls, attribute_name, attribute_value)
        return cls
    return decorator


@dataclass
class Metric(MetricInfoMixin):
    """Metric base class without LLM."""

    def __init__(
        self,
        config_name: Optional[str] = None,
        experiment_id: Optional[str] = None
    ):
        """Initialization.

        Args:
            config_name: type(string), Optional.
            experiment_id: type(string), Optional.
        """
        self._required_columns = []
        info = self._info()
        info.metric_name = camelcase_to_snakecase(self.__class__.__name__)
        info.config_name = config_name or "default"
        info.experiment_id = experiment_id or "default_experiment"
        MetricInfoMixin.__init__(self, info)

    @property
    @abstractmethod
    def name(self) -> str:
        """The metric name."""
        ...

    def _info(self) -> MetricInfo:
        """Construct the MetricInfo object. See `datasets.MetricInfo` for details.

        Warning: This function is only called once and the result is cached for all
        following .info() calls.

        Returns:
            info: (datasets.MetricInfo) The metrics information

        """
        raise NotImplementedError

    def _validate_data(self, dataset: Dataset) -> bool:
        """Validate the of the input dataset."""
        if not all(c in dataset.column_names for c in self._required_columns):
            raise ValueError("The input dataset of f{self.name} metric should include f{self._required_columns} columns.")

    def compute(
        self,
        dataset: Dataset,
        batch_size: int = None,
    ) -> Tuple[float, Dataset]:
        """Evaluate the dataset."""
        scores = []
        length = len(dataset)
        self._validate_data(dataset)
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
    def init_model(self, model: Callable):
        """This method will lazy initialize the model."""
        ...

    def parse_llm_result(self, prompts: List[str], result: LLMResult):
        """Parse the LLM Result based on the Prompt."""
        ...
