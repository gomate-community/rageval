import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import floor

from datasets import Dataset
from tqdm import tqdm


def make_batches(total_size: int, batch_size: int) -> list:
    """Take a total size and batch size and return a list of ranges for the batches."""
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
    """Metric base class without LLM."""

    batch_size: int

    @abstractmethod
    def init_model(self):
        """This method will lazy initialize the model."""
        ...

    def score(
        self,
        dataset: Dataset,
    ) -> Dataset:
        """Evaluate the dataset."""
        scores = []
        for batch in tqdm(self.get_batches(len(dataset))):
            score = self._score_batch(dataset.select(batch))
            scores.extend(score)

        return dataset.add_column(f"{self.name}", scores)  # type: ignore

    @abstractmethod
    def _score_batch(
        self,
        dataset: Dataset,
    ) -> list:
        ...

    def get_batches(self, dataset_size: int) -> list:
        """Get batches."""
        return make_batches(dataset_size, self.batch_size)


@dataclass
class MetricWithLLM(Metric):
    """Metrics based on LLM."""

    from rageval.llms.openai import OpenAILLM

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
