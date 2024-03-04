"""Base task."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Type, List

from datasets import Dataset, concatenate_datasets

from rageval.metrics import Metric


@dataclass
class BaseTask(ABC):
    """Base Task, shouldn't be used directly."""

    def __init__(self, metrics: Union[str, List[str], List[Metric]]):
        """Base task constructor."""
        self.detailed_result = []
        self.result = {}
        self.metrics = self._init_metrics(metrics)

    @property
    @abstractmethod
    def name(self) -> str:
        """The task name."""
        ...

    def _init_metrics(self, metrics):
        if not metrics:
            raise ValueError("metrics should not be empty")
        if isinstance(metrics, str):
            metrics = [metrics]
        return [self._parse_metric(m) for m in metrics]

    def _parse_metric(self, metric: Union[str, Type[Metric], Metric]):
        """
        Parse input metric in any form into a :class:`Metric` instance.

        :param metric: Input metric in any form.
        :return: A :class:`Metric` instance

        """

        if isinstance(metric, str):
            metric = metric.lower()  # ignore case

            # TODO: parse metrics in str form
            """
            for subclass in Metric.__subclasses__():
                if metric in subclass.ALIAS:
                    return subclass()
            """

        elif isinstance(metric, Metric):
            return metric
        elif issubclass(metric, Metric):
            return metric()
        else:
            raise ValueError(metric)

    def evaluate(self, testset) -> Dataset:
        """Evaluation each metrics."""

        self._validate_columns(testset)
        for m in self.metrics:
            res, de_res = m.compute(testset)
            self.result[m.name] = res
            self.detailed_result.append(de_res)
        return self.result

    def _validate_columns(self, testset: Dataset):
        """Make sure columns in testset is subset of required columns."""

        if not set(self.required_columns).issubset(set(testset.column_names)):
            print("Testset should contain following columns: ", ', '.join(self.required_columns))
            raise ValueError(testset)

    def obtain_detailed_result(self):
        """Obtain instance level result for the test case."""

        if not self.detailed_result:
            raise NameError(self.detailed_result)
        colnames = self.required_columns + [m.name for m in self.metrics]
        self.detailed_result = concatenate_datasets(self.detailed_result).select_columns(colnames)
        return self.detailed_result
