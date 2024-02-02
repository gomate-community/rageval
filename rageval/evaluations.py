from __future__ import annotations

import typing as t
import numpy as np
from dataclasses import dataclass, field
from datasets import Dataset, concatenate_datasets

from rageval.metrics import Metric


def evaluate(
    testset: Dataset,
    #task: Task,
    metrics: list[Metric] | None = None
    )-> Result:
    # Validation
    # TODO

    # run evaluation
    [m.init_model() for m in metrics]
    scores = []
    for metric in metrics:
        print(f"evaluating with [{metric.name}]")
        scores.append(metric.score(testset).select_columns(metric.name))

    # evaluation log
    # TODO

    return Result(
        scores=concatenate_datasets(scores, axis=1),
        testset=testset,
    )

@dataclass
class Result(dict):
    scores: Dataset
    testset: Dataset | None = None

    def __post_init__(self):
        values = []
        for cn in self.scores.column_names:
            value = np.nanmean(self.scores[cn])
            self[cn] = value
            if cn not in self.binary_columns:
                value = t.cast(float, value)
                values.append(value + 1e-10)

    def to_pandas(self, batch_size: int | None = None, batched: bool = False):
        if self.testset is None:
            raise ValueError("testset is not provided for the results class")
        assert self.scores.shape[0] == self.testset.shape[0]
        result_ds = concatenate_datasets([self.testset, self.scores], axis=1)

        return result_ds.to_pandas(batch_size=batch_size, batched=batched)

    def __repr__(self) -> str:
        scores = self.copy()
        score_strs = [f"'{k}': {v:0.4f}" for k, v in scores.items()]
        return "{" + ", ".join(score_strs) + "}"
