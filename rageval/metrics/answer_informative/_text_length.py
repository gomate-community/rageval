from dataclasses import dataclass
from typing import List
import numpy as np

import datasets

from rageval.metrics import Metric, add_attribute


_DESCRIPTION = """\
Textlength is a metric used to evaluate the length of a model-generated response.

It measures the number of tokens in the generated text by first converting the text into tokens and then counting the total number. This metric provides insight into the verbosity or conciseness of the model's output, offering a standardized way to compare text length across different responses.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str

Optional Args:
    None

Functions:
    _compute_one: Evaluating the length of answer.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "A",
    ...         "C",
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> tokenize_model = rl.models.Tokenizer("Qwen/Qwen2-0.5B-Instruct")
    >>> metric = rl.metrics.TextLength(tokenize_model=tokenize_model)
    >>> metric.mtype
    'answer_informative'
"""


@dataclass
@add_attribute('mtype', 'answer_informative')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TextLength(Metric):
    """Estimates the text length of answers."""

    name = "text_length"

    ALIAS = ['text_length']

    def __init__(self, tokenize_model: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Explicitly initialize TextLength.

        Ensure all parent classes are initialized.
        """
        self.tokenize_model = tokenize_model
        super().__init__()

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation="",
            homepage="",
            features=datasets.Features(
                {
                    "answers": datasets.Value("string"),
                }
            ),
            codebase_urls=[],
            reference_urls=[]
        )

    def _compute_one(
        self,
        answer: str,
    ) -> float:
        """Evaluating the text length of answer."""
        length = len(self.tokenize_model.tokenizer(answer, return_tensors="pt")['input_ids'][0])
        return length

    def _compute_batch(
        self,
        pred_answers,
    ) -> List[float]:
        """Evaluate the text length of a batch of answers."""
        results = [self._compute_one(answer) for answer in pred_answers]
        return results
