from dataclasses import dataclass
from typing import List

import datasets
import numpy as np

from rageval.metrics import Metric, add_attribute


_DESCRIPTION = """\
AnswerEMCorrectness evaluates answer correctness based on exact matching of annotated short answers.

For details, see the paper: https://arxiv.org/abs/2204.06092.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    batch_size : int, Batch size for openai completion.
    ignore_case : bool, whether to ignore case when comparing the answer and ground truth answers.

Optional Args:
    None

Functions:
    _compute_one: compute the score by measure whether the args:`answer` contains short answer in list:`gt_answers`.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "Ali Dael has the highest goals in men's world international football with 109 goals. Josef Bican has "
    ...         "the highest goals all-time in men's football and Christine Sinclair has the highest goals in women's "
    ...         "world international football.",
    ...         "A supercentenarian is someone who has reached the age of 110. Sarah Knauss, whose age is undisputed, "
    ...         "was the oldest person ever from the United States and the second-oldest fully documented person ever. "
    ...         "Jeanne Calment was a French supercentenarian and the oldest human whose age is well-documented, with "
    ...         "a lifespan of 122 years and 164 days, and was the oldest person in the world as of 1997. In 1985, "
    ...         "the oldest living person was Mathew Beard and in 1986 it was Augusta Holtz, who lived 115 years and "
    ...         "79 days, from 1871 to 1986."
    ...     ],
    ...     "gt_answers": [
    ...         [
    ...             ["Daei", "Ali Daei"],
    ...             ["Bican", "Josef Bican"],
    ...             ["Sinclair","Christine Sinclair"]
    ...         ],
    ...         [
    ...             ["Jeanne Calment"],
    ...             ["Sarah Knauss"],
    ...             ["Augusta-Holtz"],
    ...         ]
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerEMCorrectness()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> s, ds = metric.compute(dataset, batch_size=1)
    >>> assert 0 <= s <= 1
    >>> type(ds)
    <class 'datasets.arrow_dataset.Dataset'>
"""

_CITATION = """\
@misc{stelmakh2023asqa,
      title={ASQA: Factoid Questions Meet Long-Form Answers},
      author={Ivan Stelmakh and Yi Luan and Bhuwan Dhingra and Ming-Wei Chang},
      year={2023},
      eprint={2204.06092},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerEMCorrectness(Metric):
    """Estimates correctness using annotated short answers."""

    name = "answer_exact_match"

    ALIAS = ['answer_exact_match']

    def __init__(self, ignore_case: bool = False):
        """Explicitly initialize the AnswerEMCorrectness to ensure all parent class initialized."""
        super().__init__()
        self._required_columns = ['answers', 'gt_answers']
        self.ignore_case = ignore_case

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
                    "gt_answers": datasets.Sequence(datasets.Value("string"))
                }
            ),
            codebase_urls=[],
            reference_urls=["https://arxiv.org/abs/2204.06092"]
        )

    def _compute_one(self, output: str, short_answers: List[List[str]]) -> float:
        """Compute the correctness of a single answer."""
        acc = []
        if self.ignore_case:
            output = output.lower()
            short_answers = [[a.lower() for a in candidate_short_answers] for candidate_short_answers in short_answers]
        for candidate_short_answers in short_answers:
            for candidate_short_answer in candidate_short_answers:
                if candidate_short_answer in output:
                    acc.append(True)
                    break
            else:
                acc.append(False)
        return np.average(acc)

    def _compute_batch(self, dataset: datasets.Dataset) -> List[float]:
        """Compute the correctness of a batch of answers."""
        return [self._compute_one(output, short_answers)
                for output, short_answers in zip(dataset["answers"], dataset["gt_answers"])]
