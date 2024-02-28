from __future__ import annotations

from typing import Callable, List
import numpy as np
from dataclasses import dataclass, field
from datasets import Dataset, concatenate_datasets

from rageval.metrics import Metric


def evaluate(
        testset: Dataset,
        metrics: List[Metric] | None = None,
        models: List[Callable] | None = None) -> (Dataset, Dataset):
    """Conduct the evaluation on testset."""

    # run evaluation
    assert (len(metrics) == len(models))
    [metrics[i].init_model(models[i]) for i in range(len(metrics))]
    avg_scores = []
    instance_scores = [testset]
    for metric in metrics:
        print(f"evaluating with [{metric.name}]")
        avg_score, _testset = metric.compute(testset)
        avg_scores.append(Dataset.from_dict({metric.name: [avg_score]}))
        instance_scores.append(_testset.select_columns(metric.name))

    return concatenate_datasets(avg_scores), concatenate_datasets(instance_scores)
