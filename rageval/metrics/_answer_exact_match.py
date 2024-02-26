from dataclasses import dataclass
from typing import List

import numpy as np
from datasets import Dataset

from rageval.metrics import Metric, add_attribute


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
class AnswerExactMatch(Metric):
    """Estimates correctness using annotated short answers."""

    name = "answer_exact_match"
    _required_columns = ['answers', 'gt_answers']

    def _compute_one(self, output: str, short_answers: List[List]) -> float:
        """Compute the correctness of a single answer."""
        acc = []
        for candidate_short_answers in short_answers:
            for candidate_short_answer in candidate_short_answers:
                if candidate_short_answer in output:
                    acc.append(True)
                    break
            else:
                acc.append(False)
        return np.average(acc)

    def _compute_batch(self, dataset: Dataset) -> List[float]:
        """Compute the correctness of a batch of answers."""
        return [self._compute_one(output, short_answers)
                for output, short_answers in zip(dataset["answers"], dataset["gt_answers"])]
