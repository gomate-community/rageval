import re
from dataclasses import dataclass
from typing import List, Tuple
import evaluate

import datasets

from rageval.metrics import Metric, add_attribute


_DESCRIPTION = """\
BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
Scores are calculated by comparing individual translated segments, e.g., sentences, with a set of high-quality reference translations.
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
    _compute_one: compute bleu score for single prediction with its references

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "It is a guide to action which ensures that the military always obeys the commands of the party.",
    ...         "It is to insure the troops forever hearing the activity guidebook that party direct.",
    ...     ],
    ...     "gt_answers": [
    ...         [
    ...             "It is a guide to action that ensures that the military will forever heed Party commands.",
    ...             "It is the guiding principle which guarantees the military forces always being under the command of the Party.",
    ...             "It is the practical guide for the army always to heed the directions of the party.",
    ...         ],
    ...         [
    ...             "It is a guide to action that ensures that the military will forever heed Party commands.",
    ...             "It is the guiding principle which guarantees the military forces always being under the command of the Party.",
    ...             "It is the practical guide for the army always to heed the directions of the party.",
    ...         ]
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerBleuScore()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> score, results = metric.compute(dataset["answers"], dataset["gt_answers"], 1)
    >>> score
    0.3172992057845065
    >>> results[0]
    0.49697705300310346
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

    """Note: this metric is just fit for English data by now(24/03/12)"""

    name = "answer_bleu"

    ALIAS = ['answer_bleu']

    def __init__(self):
        """Explicitly initialize the AnswerBleuScore to ensure all parent class initialized."""
        super().__init__()
        self.info = evaluate.MetricInfo(
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
            codebase_urls=[
                "https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py",
                "https://github.com/huggingface/datasets/blob/main/metrics/bleu/bleu.py"
            ],
            reference_urls=["https://www.aclweb.org/anthology/P02-1040.pdf"]
        )

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"        

    def _clean_special_tokens(self, sentence: str, subword: str) -> str:
        """Clean special word in sentence"""

        sentence = sentence.strip()
        if subword is not None:
            sentence = re.sub(subword, "", sentence)
        return sentence

    def _compute_one(
        self,
        pred_answers: List[str],
        ref_answers: List[List[str]]
    ) -> List[float]:
        """Compute the bleu score of a batch of answers."""
        scores = []
        bleu = datasets.load_metric("bleu")
        for output, gt_answers in zip(pred_answers, ref_answers):
            output_clean = self._clean_special_tokens(output, None)
            predictions = [output_clean.split(' ')]
            references = []
            for gt_answer in gt_answers:
                gt_answer_clean = self._clean_special_tokens(gt_answer, None)
                references.append(list(gt_answer_clean.split(' ')))
            bleu_result = bleu.compute(predictions=predictions, references=[references])
            bleu_score = bleu_result['bleu']
            scores.append(bleu_score)

        return scores

    def compute(
        self,
        pred_answers: List[str],
        ref_answers: List[List[str]],
        batch_size: int,
    ) -> Tuple[float, List[float]]:
        """Evaluate the dataset."""

        bleu = datasets.load_metric("bleu")
        predictions = []
        references = []
        for output, gt_answers in zip(pred_answers, ref_answers):
            output_clean = self._clean_special_tokens(output, None)
            predictions.append(list(output_clean.split(' ')))
            reference = []
            for gt_answer in gt_answers:
                gt_answer_clean = self._clean_special_tokens(gt_answer, None)
                reference.append(list(gt_answer_clean.split(' ')))
            references.append(reference)
        bleu_result = bleu.compute(predictions=predictions, references=references)
        bleu_score = bleu_result['bleu']
        scores = self._compute_one(pred_answers, ref_answers)

        return bleu_score, scores
