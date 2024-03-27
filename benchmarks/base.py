from typing import List, Union, Dict, Any, Tuple, Optional
from abc import abstractmethod, ABC
from dataclasses import dataclass
# import importlib
from datasets import Dataset, load_dataset
from rageval.metrics import Metric
from .utils import save_json

class BaseBenchmark(ABC):
    """Base class for benchmarks."""

    metrics: List[Metric] = []
    dataset: Dataset

    def __init__(self, batch_size: int = 1) -> None:
        """Initialization."""
        self.batch_size = batch_size

    @property
    @abstractmethod
    def name(self) -> str:
        """The benchmark name."""
        ...

    @property
    def metric_names(self) -> List[str]:
        """The metric names."""
        return [m.name for m in self.metrics]

    def load_data(self, **kwargs) -> None:
        """Load the dataset with answers to evaluate."""
        print("Load dataset...")
        self.dataset = load_dataset(**kwargs)
        print("Dataset loaded.")

    @abstractmethod
    def _evaluate(self) -> Tuple[Dict[Any, Any], Dataset]:
        """Evaluate the dataset and return the results and the detailed dataset with each sample scores."""
        ...

    def evaluate(self, **kwargs) -> Dict[Any, Any]:
        """Load datasets and evaluate it, return a result dict."""
        if not hasattr(self, "dataset"):
            self.load_data(**kwargs)
        print("Start evaluating...")
        self.results, self.dataset = self._evaluate()
        print("Evaluation finished.")
        return self.results

    def set_metric(self, metrics: List[Metric]) -> None:
        """Reset the metrics."""
        if all(isinstance(m, Metric) for m in metrics):
            self.metrics = metrics
        else:
            raise ValueError("The metrics should be a list of Metric objects.")

    def save_dataset(self, file_path: str) -> None:
        """Save the result to files."""
        if not hasattr(self, "dataset"):
            raise ValueError("Please load the dataset and evaluate it first.")
        self.dataset.to_json(file_path, orient="records")
        print(f"Dataset saved to {file_path}.")

    def save_results(self, file_path: str) -> None:
        """Save the result to files."""
        if not hasattr(self, "results"):
            raise ValueError("Please run evaluation first.")
        save_json(self.results, file_path)
        print(f"Results saved to {file_path}.")

    # def get_metric(self, name: str, **kwargs) -> Union[Metric, MetricWithLLM]:
    #     """Get the metric by name."""
    #     module = importlib.import_module(f"rageval.metrics")
    #     return getattr(module, name)(**kwargs)
