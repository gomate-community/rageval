from typing import List, Tuple, Callable, Optional, Iterable
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
    There are four types, i.e., 'AnswerCorrectness', 'AnswerGroundedness', 'ContextRelevancy', 'ContextAdequacy', \
    for all RAG metrics.
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

    def _validate_data(
        self,
        predictions: Optional[Iterable] = None,
        references: Optional[Iterable] = None,
        *args: Optional[Iterable]
    ) -> None:
        """Validate the of the input dataset."""
        if len(predictions) != len(references) or any(len(predictions) != len(arg) for arg in args):
            raise ValueError("The length of predictions and references should be the same.")

    def compute(
        self,
        batch_size: int = None,
        predictions: Optional[Iterable] = None,
        references: Optional[Iterable] = None,
        *args: Optional[Iterable],
    ) -> Tuple[float, List[float]]:
        """Evaluate the dataset."""
        self._validate_data(predictions, references, *args)
        scores = []
        length = len(predictions)
        if batch_size:
            for start in tqdm(range(0, length, batch_size)):
                end = start + batch_size
                end = end if end < length else length
                score = self._compute_batch(
                    predictions[start:end],
                    references[start:end],
                    *[arg[start:end] for arg in args],
                )
                scores.extend(score)
        else:
            scores = self._compute_batch(predictions, references, *args)

        return np.average(scores), scores

    @abstractmethod
    def _compute_batch(
        self,
        predictions: Optional[Iterable] = None,
        references: Optional[Iterable] = None,
        *args: Optional[Iterable]
    ) -> List[float]:
        ...


@dataclass
class MetricWithLLM(Metric):
    """Metrics based on LLM."""

    def __init__(self, model: Callable):
        """Initialization."""
        super().__init__()
        self.llm = model

    @abstractmethod
    def parse_llm_result(self, prompts: List[str], result: LLMResult):
        """Parse the LLM Result based on the Prompt."""
        ...
