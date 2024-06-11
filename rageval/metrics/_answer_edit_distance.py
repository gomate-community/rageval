from dataclasses import dataclass
from typing import List
import datasets

from rageval.metrics import Metric, add_attribute

_DESCRIPTION = """\
The AnswerEditDistance is to measure the similarity between answer and gt_answer by calculating the edit distance.

This is a very traditional method, but to this day, some work is still being carried out using it, such as \
https://ieeexplore.ieee.org/abstract/document/10172590.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    batch_size : int, Batch size for openai completion.

Optional Args:
    None

Functions:
    _compute_one: evaluating the similarity between answer and gt_answer by calculating the edit distance.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "Language models trained on massive code corpora can generalize to tasks without the need "
    ...         "for task-specific fine tuning."
    ...     ],
    ...     "gt_answers": [
    ...         "Large language models trained on massive code corpora can generalize to new tasks without the need "
    ...         "for task-specific fine-tuning."
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerEditDistance()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)
    >>> assert score == 5 / 18
"""

_CITATION = """\
@INPROCEEDINGS{10172590,
    author={Nashid, Noor and Sintaha, Mifta and Mesbah, Ali},
    booktitle={2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE)},
    title={Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning},
    year={2023},
    volume={},
    number={},
    pages={2450-2462},
    doi={10.1109/ICSE48619.2023.00205}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerEditDistance(Metric):
    """Estimates the similarity between answers and gt_answers."""

    name = "answer_edit_distance"

    ALIAS = ['answer_edit_distance']

    def __init__(self):
        """
        Explicitly initialize AnswerEditDistance.

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
            homepage="",
            features=datasets.Features(
                {
                    "answers": datasets.Value("string"),
                    "gt_answers": datasets.Value("string")
                }
            ),
            codebase_urls=[],
            reference_urls=["https://ieeexplore.ieee.org/abstract/document/10172590"]
        )

    def _compute_one(
        self,
        pred_answer: str,
        ref_answer: str
    ) -> float:
        """Evaluating the similarity between answer and gt_answer by calculating the edit distance."""
        pred_answer = pred_answer.split()
        ref_answer = ref_answer.split()
        m, n = len(pred_answer), len(ref_answer)

        if m == 0 or n == 0:
            return 0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
                if pred_answer[i - 1] != ref_answer[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1)
                else:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])

        return dp[m][n] / m

    def _compute_batch(
        self,
        pred_answers: List[str],
        ref_answers: List[str],
    ) -> List[float]:
        """Evaluate the similarity of a batch of answers."""
        return [
            self._compute_one(pred_answer, reference)
            for pred_answer, reference in zip(pred_answers, ref_answers)
        ]
