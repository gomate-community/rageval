from dataclasses import dataclass
from typing import List, Tuple

import datasets
from sacrebleu.metrics import CHRF
import numpy as np

from rageval.metrics import Metric, add_attribute


_DESCRIPTION = """\
    ChrF and ChrF++ are two MT evaluation metrics. They both use the F-score statistic for character n-gram matches, and ChrF++ adds word n-grams as well which correlates more strongly with direct assessment.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    predictions (list of str): The predicted sentences.
    references (list of list of str): The references. There should be one reference sub-list for each prediction sentence.
    char_order (int): Character n-gram order. Defaults to `6`.
    word_order (int): Word n-gram order. If equals to `2`, the metric is referred to as chrF++. Defaults to `0`.
    beta (int): Determine the importance of recall w.r.t precision. Defaults to `2`.
    lowercase (bool): if `True`, enables case-insensitivity. Defaults to `False`.
    whitespace (bool): If `True`, include whitespaces when extracting character n-grams.
    eps_smoothing (bool): If `True`, applies epsilon smoothing similar

Optional Args:
    None

Functions:
    _validate_data: validate the dataset format.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "The relationship between cats and dogs is not exactly friendly.",
    ...         "a good bookshop is just a genteel black hole that knows how to read."
    ...     ],
    ...     "gt_answers": [
    ...         ["The relationship between dogs and cats is not exactly friendly.", ],
    ...         ["A good bookshop is just a genteel Black Hole that knows how to read."]
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerCHRFCorrectness()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)
    >>> score
    84.64214891738334
    >>> results[0]
    84.41131092011067
"""

_CITATION = """\
@inproceedings{popovic-2015-chrf,
    title = "chr{F}: character n-gram {F}-score for automatic {MT} evaluation",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Tenth Workshop on Statistical Machine Translation",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W15-3049",
    doi = "10.18653/v1/W15-3049",
    pages = "392--395",
}
@inproceedings{popovic-2017-chrf,
    title = "chr{F}++: words helping character n-grams",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Second Conference on Machine Translation",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-4770",
    doi = "10.18653/v1/W17-4770",
    pages = "612--618",
}
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerCHRFCorrectness(Metric):
    """Estimates the CHRF between answers and ground truth answers."""

    name = "answer_chrf"

    ALIAS = ['answer_chrf']

    def __init__(
            self,
            char_order: int = 6,
            word_order: int = 0,
            beta: int = 2,
            lowercase: bool = False,
            whitespace: bool = False,
            eps_smoothing: bool = False
    ):
        """
        Explicitly initialize AnswerCHRFCorrectness.

        Ensure all parent classes are initialized.
        """
        super().__init__()

        self.chrf = CHRF(
            char_order=char_order,
            word_order=word_order,
            beta=beta,
            lowercase=lowercase,
            whitespace=whitespace,
            eps_smoothing=eps_smoothing
        )

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
            codebase_urls=["https://github.com/mjpost/sacreBLEU#chrf--chrf"],
            reference_urls=[
                "https://aclanthology.org/W15-3049.pdf",
                "https://aclanthology.org/W17-4770",
                "https://www.aclweb.org/anthology/W18-6319"
            ]
        )

    def _validate_data(
        self,
        pred_answers: List[str],
        ref_answers: List[List[str]]
    ) -> None:
        """Validate the input dataset."""
        super()._validate_data(pred_answers, ref_answers)
        if not all(isinstance(answer, str) for answer in pred_answers):
            raise ValueError("The type of pred_answers should be a string.")
        if not all(isinstance(a, list) and all(isinstance(item, str) for item in a) for a in ref_answers):
            raise ValueError("The type of ref_answers should be a list of strings.")

    def _compute_one(
        self,
        pred_answer: str,
        ref_answers: List[str]
    ) -> float:
        """Compute the metric for a single sentence against a single (or multiple) reference(s)."""
        # return self.chrf.sentence_score(pred_answer, ref_answers).score
        pass

    def _compute_batch(
        self,
        pred_answers: List[str],
        ref_answers: List[List[str]]
    ) -> List[float]:
        """Compute the metric for a batch of sentences against their references."""
        ref_answers = np.array(ref_answers)
        ref_answers = ref_answers.T            
        return self.chrf.corpus_score(pred_answers, ref_answers).score
