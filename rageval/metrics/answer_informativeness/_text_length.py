from dataclasses import dataclass
from typing import Optional, Iterable
from transformers import AutoTokenizer


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
    >>> metric = TextLength(tokenize_model="Qwen/Qwen2-0.5B-Instruct")
    >>> metric.mtype
    'answer_informativeness'
"""


@dataclass
@add_attribute('mtype', 'answer_informativeness')
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
        self.tokenizer = AutoTokenizer.from_pretrained(tokenize_model)
        super().__init__()

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}" # pragma: no cover

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
        *args: Optional[Iterable],
    ) -> float:
        """Evaluating the text length of answer."""
        length = len(self.tokenizer(answer, return_tensors="pt")['input_ids'][0])
        return length
