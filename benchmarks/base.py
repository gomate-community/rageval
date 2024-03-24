from typing import List, Union, Dict, Any, Tuple, Optional
from abc import abstractmethod, ABC
from dataclasses import dataclass
# import importlib
from datasets import Dataset, load_dataset
from rageval.metrics import Metric

class BaseBenchmark(ABC):
    """Base class for benchmarks."""

    metrics: List[Metric] = []
    dataset: Dataset

    def __init__() -> None:
        """Initialization."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """The benchmark name."""
        ...

    @property
    def metric_names(self) -> List[str]:
        """The metric names."""
        return [m.name for m in self.metrics]

    @abstractmethod
    def load_data(self, **kwargs) -> None:
        """Load the dataset with answers to evaluate."""
        self.dataset = load_dataset(**kwargs)

    @abstractmethod
    def _evaluate(self) -> Tuple[Dict[Any], Dataset]:
        """Evaluate the dataset and return the results and the detailed dataset with each sample scores."""
        ...

    def prepare_data(self, input_column: str, label_column: Optional[str], **kwargs) -> Dataset:
        """Prepare the dataset for different metric.

        Args:
            input_column: The column name of the input text that has already existed in self.dataset, e.g. `long_answer`.
            label_column: The column name of the label text that the metric requires, e.g. `gt_answer`.
        """
        if input_column not in self.dataset.column_names:
            raise ValueError(f"The input column {input_column} is not in the dataset. Please check the column names.")

        if not label_column:
            return self.dataset
        else:
            return self.dataset.add_column(label_column, self.dataset[input_column])

    def cal(metric: Metric, dataset: Dataset, batch_size: int = None) -> Tuple[float, Dataset]:
        """Calculate the metric score."""
        metric= {
            "name": "AnswerRougeCorrectness",
            "rouge_type": "rougeL",
            "column": "long_answer"
        }
        
        score, ds = metric.compute(dataset, batch_size)

    def evaluate(self, **kwargs) -> Dict[Any]:
        """Load datasets and evaluate it, return a result dict."""
        self.load_data(**kwargs)
        self.results, self.dataset = self._evaluate()
        return self.results

    def set_metric(self, metrics: Union[List[str], List[Metric]]) -> None:
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
        self.results.to_json(file_path, orient="records")
        print(f"Results saved to {file_path}.")

    # def get_metric(self, name: str, **kwargs) -> Union[Metric, MetricWithLLM]:
    #     """Get the metric by name."""
    #     module = importlib.import_module(f"rageval.metrics")
    #     return getattr(module, name)(**kwargs)
