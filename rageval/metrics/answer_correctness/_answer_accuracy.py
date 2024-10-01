from dataclasses import dataclass
from typing import List
import evaluate

import datasets

from rageval.metrics import Metric, add_attribute


_DESCRIPTION = """\
The AnswerAccuracy is to measure the correctness of answers.

This metric is applicable in scenarios where the LLM is required to output a unique short answer, such as options for \
multiple-choice questions or a single entity name.
The renowned MMLU dataset utilizes this metric for evaluation. In the evaluation of the MMLU dataset, probabilities \
for each answer are first calculated, and the answer with the highest probability is selected as the predicted result.
In our tool, we assume that the prediction result has already been obtained, and only perform the final score \
calculation.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    batch_size : int, Batch size for openai completion.

Optional Args:
    None

Functions:
    _compute_one: Evaluating the correctness of answer.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "A",
    ...         "B",
    ...         "C"
    ...     ],
    ...     "gt_answers": [
    ...         "A",
    ...         "C",
    ...         "C"
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerAccuracy()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> score, results = metric.compute(dataset["answers"], dataset["gt_answers"], 1)
    >>> score
    0.6666666666666666
    >>> results[0]
    True
"""

_CITATION = """\
@misc{hendrycks2021measuring,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      year={2021},
      eprint={2009.03300},
      archivePrefix={arXiv},
      primaryClass={cs.CY}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerAccuracy(Metric):
    """Estimates the correctness of answers."""

    name = "answer_accuracy"

    ALIAS = ['answer_accuracy']

    def __init__(self):
        """
        Explicitly initialize AnswerAccuracy.

        Ensure all parent classes are initialized.
        """
        super().__init__()
        self.info = evaluate.MetricInfo(
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
            codebase_urls=["https://github.com/hendrycks/test"],
            reference_urls=["https://arxiv.org/abs/2009.03300"]
        )

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _compute_one(
        self,
        answer: str,
        gt_answer: str
    ) -> float:
        """Evaluating the correctness of answer."""
        return answer == gt_answer
