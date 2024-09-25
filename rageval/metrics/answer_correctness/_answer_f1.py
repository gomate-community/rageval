import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Iterable, Union

import datasets
import numpy as np
import jieba

from rageval.metrics import Metric, add_attribute


_DESCRIPTION = """\
    F1 score combines precision and recall into a single score using their harmonic mean.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    normalize : bool, default is True, whether to normalize the text. If False, the text will be treated as a list of tokens.
    language : str, default is 'en', the language of the text. Supported languages are 'en' and 'zh'.

Optional Args:
    None

Functions:
    _normalize_text: normalize the text by removing articles, white spaces, punctuations and lowercasing.
    _validate_data: validate the dataset format.
    _f1_score: compute the f1 score between `pred` tokens and `ref` tokens.
    _compute_one: evaluate the f1 score of between `answer` and `gt_answers`, return the highest score in all pairs.

Examples:
    English:
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
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'])
    >>> round(score, 2)
    0.18

    Chinese:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "督邮，中国古代职官名，自汉代开始设置。",
    ...         "魏晋",
    ...         "北齐只设于清都郡。",
    ...         "隋代",
    ...     ],
    ...     "gt_answers": [
    ...         ["督邮，中国古代职官名，自汉代开始设置。"],
    ...         ["魏晋", "魏晋时期"],
    ...         ["北齐只设于清都郡。", "清都郡"],
    ...         ["隋代", "隋朝"]
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerF1Correctness(language='zh')
    >>> metric.mtype
    'AnswerCorrectness'
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'])
    >>> round(score, 2)
    1.0

    Other Iterables:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [[1,2,3], [4,5,6]],
    ...     "gt_answers": [[2,3,4,5,6], [1,2,3,4,5]]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerF1Correctness(normalize=False)
    >>> metric.mtype
    'AnswerCorrectness'
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'])
    >>> round(score, 2)
    0.5

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

    def __init__(self, normalize: bool = True, language: Optional[str] = "en"):
        """
        Explicitly initialize AnswerF1Correctness.

        Ensure all parent classes are initialized.
        """
        super().__init__()
        self.normalize = normalize
        self.language = language

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

    def _normalize_text(self, s: str) -> List[str]:
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return remove_articles(remove_punc(lower(s))).split()

    def _normalize_text_zh(self, s: str) -> str:
        """Normalize Chinese text."""
        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation) | {'，', '。', '？', '！', '：', '；', '“', '”', '‘', '’', '（', '）', '《', '》', '——', '……', '、'}
            return ''.join(ch for ch in text if ch not in exclude)

        return white_space_fix(remove_punc(s))

    def _f1_score(self, pred: Iterable, ref: Iterable) -> float:
        """Compute the f1 score between pred and ref."""
        pred_counter = Counter(pred)
        ref_counter = Counter(ref)

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
        pred_answer: Union[str, Iterable],
        ref_answers: Union[List[str], Iterable]
    ) -> float:
        """Evaluate the f1 score of an answer."""
        if self.normalize:
            # str, List[str] -> List[str], List[List[str]]
            if self.language == "en":
                preds = self._normalize_text(pred_answer)
                refs = [self._normalize_text(ref_answer) for ref_answer in ref_answers]
            elif self.language == "zh":
                preds = list(jieba.cut(self._normalize_text_zh(pred_answer)))
                refs = [list(jieba.cut(self._normalize_text_zh(ref_answer))) for ref_answer in ref_answers]
            else:
                raise Exception('Unsupported language: {}'.format(self.language)) # pragma: no cover
            scores = [self._f1_score(preds, ref) for ref in refs]
        else:
            scores = self._f1_score(pred_answer, ref_answers)

        return np.max(scores)
