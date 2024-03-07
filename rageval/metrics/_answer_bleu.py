from dataclasses import dataclass
from typing import List, Tuple

import datasets
from datasets import Dataset
import re

from rageval.metrics import Metric, add_attribute


_DESCRIPTION = """\
BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations.
Those scores are then averaged over the whole corpus to reach an estimate of the translation's overall quality.
Neither intelligibility nor grammatical correctness are not taken into account.

For details, see the paper: http://www.aclweb.org/anthology/P02-1040.pdf
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    batch_size : int, Batch size for openai completion.

Optional Args:
    None

Functions:
    _clean: clean special word in sentence.
    _compute_single: compute bleu score for single prediction with its references

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "Hello, world!",
    ...         "I am a metric named bleu."
    ...     ],
    ...     "gt_answers": [
    ...         [
    ...             "Hello, my world!",
    ...         ],
    ...         [
    ...             "I am a metric named bleu.",
    ...             "I am bleu metric.",
    ...         ]
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerBleuScore()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> s, ds = metric.compute(dataset, batch_size=1)
    >>> assert 0 <= s <= 1
    >>> type(ds)
    <class 'datasets.arrow_dataset.Dataset'>

"""


_CITATION = """\
@misc{Kishore2002bleu,
      title={Bleu: a method for automatic evaluation of machine translation},
      author={Kishore Papineni and Salim Roukos and Todd Ward and Wei-Jing Zhu},
      year={2002},
      page={311-318},
      primaryClass={cs.CL}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerBleuScore(Metric):
    """Bleu score computing with good quality reference."""

    name = "answer_bleu"

    ALIAS = ['answer_bleu']

    def __init__(self):
        """Explicitly initialize the AnswerBleuScore to ensure all parent class initialized."""
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
                    "gt_answers": datasets.Sequence(datasets.Value("string"))
                }
            ),
            codebase_urls=[],
            reference_urls=["http://www.aclweb.org/anthology/P02-1040.pdf"]
        )

    def _clean(self, sentence: str, subword: str) -> str:
        """Clean special word in sentence"""

        sentence = sentence.strip()
        if subword is not None:
            sentence = re.sub(subword, "", sentence)
        return sentence

    def _compute_single(self, dataset: datasets.Dataset) -> List[float]:
        """Compute the bleu score of a batch of answers."""

        scores = []
        bleu = datasets.load_metric("bleu")
        for output, gt_answers in zip(dataset["answers"], dataset["gt_answers"]):
            output_clean = self._clean(output, None)
            predictions = []
            predictions.append(output_clean.split(' '))
            references = []
            for gt_answer in gt_answers:
                gt_answer_clean = self._clean(gt_answer, None)
                reference = []
                reference.append(gt_answer_clean.split(' '))
            references.append(reference)
            bleu_result = bleu.compute(predictions=predictions, references=references)
            bleu_score = bleu_result['bleu']
            scores.append(bleu_score)

        return scores

    def compute(
        self,
        dataset: Dataset,
        batch_size: int = None,
    ) -> Tuple[float, Dataset]:
        """Evaluate the dataset."""

        bleu = datasets.load_metric("bleu")
        predictions = []
        references = []
        reference = []
        for output, gt_answers in zip(dataset["answers"], dataset["gt_answers"]):
            output_clean = self._clean(output, None)
            predictions.append(list(output_clean.split(' ')))
            for gt_answer in gt_answers:
                gt_answer_clean = self._clean(gt_answer, None)
                reference.append(list(gt_answer_clean.split(' ')))
            references.append(reference)
        bleu_result = bleu.compute(predictions=predictions, references=references)
        bleu_score = bleu_result['bleu']
        scores = self._compute_single(dataset)

        return bleu_score, dataset.add_column(f"{self.name}", scores)

    def _compute_batch(self, dataset: Dataset) -> list:
        pass
