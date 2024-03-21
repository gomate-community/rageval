from typing import List, Union
from abc import abstractmethod, ABC
from dataclasses import dataclass
import importlib
from datasets import Dataset
from rageval.metrics import Metric, MetricWithLLM

class BaseBenchmark(ABC):
    """Base class for benchmarks."""

    metrics: List[str]

    def __init__() -> None:
        """Initialization."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """The benchmark name."""
        ...

    @abstractmethod
    def load_data(self,) -> Dataset:
        """Load the dataset with answers to evaluate."""
        ...

    @abstractmethod
    def evaluate(self,) -> Dataset:
        """Evaluate the dataset and return the dataset with scores."""
        ...
    
    @abstractmethod
    def save_result(self,) -> None:
        """Save the result to files."""
        ...

    def get_metric(self, name: str, **kwargs) -> Union[Metric, MetricWithLLM]:
        """Get the metric by name."""
        module = importlib.import_module(f"rageval.metrics")
        return getattr(module, name)(**kwargs)
