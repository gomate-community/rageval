from dataclasses import dataclass
from typing import List, Tuple

import datasets
from datasets import Dataset

from rageval.metrics import Metric, add_attribute

_DESCRIPTION = """\
    TER (Translation Edit Rate, also called Translation Error Rate) is a metric to quantify the edit operations that a
hypothesis requires to match a reference translation. The implementation is already present in sacrebleu
(https://github.com/mjpost/sacreBLEU#ter), which in turn is inspired by the TERCOM implementation, which can be found
here: https://github.com/jhclark/tercom.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    normalized (boolean): If `True`, applies basic tokenization and normalization to sentences. Defaults to `False`.
    ignore_punct (boolean): If `True`, applies basic tokenization and normalization to sentences. Defaults to `False`.
    support_zh_ja_chars (boolean): If `True`, tokenization/normalization supports processing of Chinese characters,
                                    as well as Japanese Kanji, Hiragana, Katakana, and Phonetic Extensions of Katakana.
                                    Only applies if `normalized = True`. Defaults to `False`.
    case_sensitive (boolean): If `False`, makes all predictions and references lowercase to ignore differences in case. Defaults to `False`.

Optional Args:
    None

Functions:
    _validate_data: validate the dataset format.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "does this sentence match??",
    ...         "what about this sentence?",
    ...         "What did the TER metric user say to the developer?"
    ...     ],
    ...     "gt_answers": [
    ...         ["does this sentence match", "does this sentence match!?!"],
    ...         ["wHaT aBoUt ThIs SeNtEnCe?", "wHaT aBoUt ThIs SeNtEnCe?"],
    ...         ["Your jokes are...", "...TERrible"]
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerTERCorrectness()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)
"""

_CITATION = """\
@inproceedings{snover-etal-2006-study,
    title = "A Study of Translation Edit Rate with Targeted Human Annotation",
    author = "Snover, Matthew  and
      Dorr, Bonnie  and
      Schwartz, Rich  and
      Micciulla, Linnea  and
      Makhoul, John",
    booktitle = "Proceedings of the 7th Conference of the Association for Machine Translation in the Americas: Technical Papers",
    month = aug # " 8-12",
    year = "2006",
    address = "Cambridge, Massachusetts, USA",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2006.amta-papers.25",
    pages = "223--231",
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
class AnswerTERCorrectness(Metric):
    """Estimates the TER between answers and ground truth answers."""

    name = "answer_ter"

    ALIAS = ['answer_ter']

    def __init__(
        self,
        normalized: bool = False,
        ignore_punct: bool = False,
        support_zh_ja_chars: bool = False,
        case_sensitive: bool = False
    ):
        """
        Explicitly initialize AnswerTERCorrectness.

        Ensure all parent classes are initialized.
        """
        super().__init__()
        self.normalized = normalized
        self.ignore_punct = ignore_punct
        self.support_zh_ja_chars = support_zh_ja_chars
        self.case_sensitive = case_sensitive

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
            codebase_urls=["https://github.com/huggingface/datasets/blob/main/metrics/ter/ter.py"],
            reference_urls=["https://aclanthology.org/2006.amta-papers.25", "https://www.aclweb.org/anthology/W18-6319"]
        )

    def _validate_data(
        self,
        pred_answers: List[str],
        ref_answers: List[List[str]]
    ) -> None:
        """Validate the input predictions and references."""
        if not all(isinstance(pred_answer, str) for pred_answer in pred_answers):
            raise ValueError("The type of pred_answers should be a list of strings.")
        if not all(isinstance(reference_list, list) and all(isinstance(reference, str) for reference in reference_list) for reference_list in ref_answers):
            raise ValueError("The type of ref_answers should be a list of lists of strings.")

    def compute(
        self,
        pred_answers: List[str],
        ref_answers: List[List[str]],
        batch_size: int,
    ) -> Tuple[float, Dataset]:
        """Evaluate the dataset."""
        ter = datasets.load_metric("ter")
        result = ter.compute(
            predictions=pred_answers,
            references=ref_answers,
            normalized=self.normalized,
            ignore_punct=self.ignore_punct,
            support_zh_ja_chars=self.support_zh_ja_chars,
            case_sensitive=self.case_sensitive
        )
        scores = [
            ter.compute(
                predictions=[pred_answers[i]],
                references=[ref_answers[i]],
                normalized=self.normalized,
                ignore_punct=self.ignore_punct,
                support_zh_ja_chars=self.support_zh_ja_chars,
                case_sensitive=self.case_sensitive
            )['score']
            for i in range(len(pred_answers))
        ]
        return result, scores

    def _compute_batch(
        self,
        pred_answers: List[str],
        ref_answers: List[List[str]]
    ) -> list:
        pass
