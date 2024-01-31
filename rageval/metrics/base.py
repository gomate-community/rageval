import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import floor

from datasets import Dataset
from tqdm import tqdm

from rageval.llms import llm_factory
from rageval.llms import ragevalLLM



def make_batches(total_size: int, batch_size: int) -> list[range]:
    """
    Take a total size and batch size and return a list of ranges for the batches
    """
    tail = total_size % batch_size
    num_batches = floor(total_size / batch_size)
    batches = [
        range(i, i + batch_size) for i in range(0, batch_size * num_batches, batch_size)
    ]
    if tail != 0:
        batches.append(range(batch_size * num_batches, batch_size * num_batches + tail))

    return batches



@dataclass
class Metric(ABC):
    """Metric base class without LLM"""
    batch_size: int

    @abstractmethod
    def init_model(self):
        """
        This method will lazy initialize the model.
        """
        ...
    
    def score(
        self,
        dataset: Dataset,
    ) -> Dataset:
        scores = []
        for batch in tqdm(self.get_batches(len(dataset))):
            score = self._score_batch(dataset.select(batch))
            scores.extend(score)

        return dataset.add_column(f"{self.name}", scores)  # type: ignore

    @abstractmethod
    def _score_batch(
        selfself,
        dataset: Dataset,
    ) -> list:
        ...

    def get_batches(self, dataset_size: int) -> list[range]:
        return make_batches(dataset_size, self.batch_size)


@dataclass
class MetricWithLLM(Metric):
    llm: ragevalLLM = field(default_factory=llm_factory)

    def init_model(self):
        """
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
