from typing import List, Tuple, Callable, Optional, Iterable
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
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
class Metric():
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
        """  # pragma: no cover

    @property
    @abstractmethod
    def name(self) -> str:
        """The metric name."""
        ...  # pragma: no cover

    def _validate_data(
        self,
        pred_answers: Optional[Iterable] = None,
        ref_answers: Optional[Iterable] = None,
        *args: Optional[Iterable]
    ) -> None:
        """Validate the of the input dataset."""
        if (pred_answers and ref_answers):
            if len(pred_answers) != len(ref_answers) or any(len(pred_answers) != len(arg) for arg in args):
                raise ValueError("The length of predictions and references should be the same.")  # pragma: no cover

    def compute(
        self,
        pred_answers: Optional[Iterable] = None,
        ref_answers: Optional[Iterable] = None,
        batch_size: Optional[int] = None,
        *args: Optional[Iterable],
    ) -> Tuple[float, List[float]]:
        """
        Evaluate the dataset.

        Return average scores of all inputs and a score list for each example.
        """
        self._validate_data(pred_answers, ref_answers, *args)
        scores = self._compute_batch(pred_answers, ref_answers, *args)

        return np.average(scores), scores

    @abstractmethod
    def _compute_one(
        self,
        pred_answer: Optional[Iterable] = None,
        ref_answer: Optional[Iterable] = None,
        *args: Optional[Iterable]
    ) -> float:
        ...  # pragma: no cover

    def _compute_batch(
        self,
        pred_answers: Optional[Iterable] = None,
        ref_answers: Optional[Iterable] = None,
        *args: Optional[Iterable]
    ) -> List[float]:
        """Compute the metric for a batch of predictions and references."""
        scores = []
        if (pred_answers and ref_answers):  # if both columns exist
            for pred_answer, ref_answer in tqdm(zip(pred_answers, ref_answers),
                                                desc=f"Computing {self.name}",
                                                total=len(pred_answers)):
                scores.append(self._compute_one(pred_answer, ref_answer))
        else:
            for pred_answer in tqdm(pred_answers,
                                    desc=f"Computing {self.name}",
                                    total=len(pred_answers)):
                scores.append(self._compute_one(pred_answer))
        return scores


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
        ...  # pragma: no cover
