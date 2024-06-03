import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import List

import datasets
import numpy as np

from rageval.metrics import Metric, add_attribute

_DESCRIPTION = """\
    F1 score combines precision and recall into a single score using their harmonic mean.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str

Optional Args:
    None

Functions:
    _normalize_text: normalize the text by removing articles, white spaces, punctuations and lowercasing.
    _validate_data: validate the dataset format.
    _f1_score: compute the f1 score between `pred` string and `ref` string.
    _compute_one: evaluate the f1 score of between `answer` and `gt_answers`, return the highest score in all pairs.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "Democrat rick kriseman won the 2016 mayoral election, while re- publican former mayor rick baker did so in the 2017 mayoral election."
    ...     ],
    ...     "gt_answers": [
    ...         [
    ...             "Kriseman",
    ...             "Rick Kriseman"
    ...         ]
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerF1Correctness()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> s, ds = metric.compute(dataset, batch_size=1)
    >>> assert 0 <= s <= 1
    >>> type(ds)
    <class 'datasets.arrow_dataset.Dataset'>
"""

_CITATION = """\

"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerF1Correctness(Metric):
    """Estimates the F1 between answers and ground truth answers."""

    name = "answer_f1"

    ALIAS = ['answer_f1']

    def __init__(self):
        """
        Explicitly initialize AnswerF1Correctness.

        Ensure all parent classes are initialized.
        """
        super().__init__()
        self._required_columns = ['answers', 'gt_answers']

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "answers": datasets.Value("string"),
                    "gt_answers": datasets.Sequence(datasets.Value("string"))
                }
            ),
            codebase_urls=[],
            reference_urls=[]
        )

    def _normalize_text(self, s: str) -> str:
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _validate_data(
        self, 
        predictions: List[str],
        references: List[List[str]]
    ) -> None:
        """
        Validate the of the input dataset.
        Args:
            predictions (List[str]): A list of predicted answers.
            references (List[List[str]]): A list of lists, each containing reference answers.

        Raises:
            ValueError: If the predictions or references are not in the correct format.
        """
        if not all(isinstance(prediction, str) for prediction in predictions):
            raise ValueError("The type of predictions should be a list of strings.")
        if not all(isinstance(ref, list) and all(isinstance(item, str) for item in ref) for ref in references):
            raise ValueError("The type of references should be a list of lists of strings.")

    def _f1_score(self, pred: str, ref: str) -> float:
        """Compute the f1 score between pred and ref."""
        normalized_prediction = self._normalize_text(pred)
        normalized_ground_truth = self._normalize_text(ref)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()

        pred_counter = Counter(prediction_tokens)
        ref_counter = Counter(ground_truth_tokens)

        tp = sum((pred_counter & ref_counter).values())
        fp = sum((pred_counter - ref_counter).values())
        fn = sum((ref_counter - pred_counter).values())

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1

        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def _compute_one(
        self,
        answer: str,
        gt_answers: List[str]
    ) -> float:
        """Evaluate the f1 score of an answer."""
        scores = []
        for gt_answer in gt_answers:
            score = self._f1_score(answer, gt_answer)
            scores.append(score)

        return np.max(scores)

    def _compute_batch(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> List[float]:
        """Evaluate the f1 score of a batch of answers."""
        return [self._compute_one(prediction, ref) for prediction, ref in zip(predictions, references)]
