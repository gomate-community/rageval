import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

import datasets
import numpy as np
import spacy

from rageval.metrics import Metric, add_attribute

_DESCRIPTION = """\
The Disambig-F1 is a variant of the F1 score, estimates the similarity between the disambiguation of the answer and the ground truth answer.

The original metric was presented in [ASQA paper](https://aclanthology.org/2022.emnlp-main.566/), and implemented through [this code](https://github.com/google-research/language/blob/master/language/asqa/scoring.py#L273). And we adopted an [alternative implementation](https://github.com/jzbjyb/FLARE/tree/main/src/datasets.py#L29) from the paper [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983).
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    model : str, model name of spacy model to ner.

Optional Args:
    None

Functions:
    _normalize_text: normalize the text by removing articles, white spaces, punctuations and lowercasing.
    _ner: extract named entities from the text.
    _validate_data: validate the dataset format.
    _f1_score: compute the f1 score between `pred` string and `ref` string.
    _compute_one: evaluate the disambig f1 score of between `answer` and `gt_answers`, return the highest score in all pairs.

Examples:
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
    >>> metric = rl.metrics.AnswerDisambigF1Correctness(model="en_core_web_sm")
    >>> metric.mtype
    'AnswerCorrectness'
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)
    >>> assert 0 <= score <= 1
"""

_CITATION = """\
@inproceedings{stelmakh-etal-2022-asqa,
    title = "{ASQA}: Factoid Questions Meet Long-Form Answers",
    author = "Stelmakh, Ivan  and
      Luan, Yi  and
      Dhingra, Bhuwan  and
      Chang, Ming-Wei",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.566",
    doi = "10.18653/v1/2022.emnlp-main.566",
    pages = "8273--8288",
}
@misc{jiang2023active,
      title={Active Retrieval Augmented Generation},
      author={Zhengbao Jiang and Frank F. Xu and Luyu Gao and Zhiqing Sun and Qian Liu and Jane Dwivedi-Yu and Yiming Yang and Jamie Callan and Graham Neubig},
      year={2023},
      eprint={2305.06983},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerDisambigF1Correctness(Metric):
    """Estimates the Disambig-F1 between answers and ground truth answers."""

    name = "answer_disambig_f1"

    ALIAS = ['answer_disambig_f1']

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Explicitly initialize AnswerDisambigF1Correctness.

        Ensure all parent classes are initialized.
        Ensure spacy ner model is initialized.
        """
        super().__init__()
        self.model = model
        self.nlp = spacy.load(model)

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
            codebase_urls=[
                "https://github.com/google-research/language/blob/master/language/asqa",
                "https://github.com/jzbjyb/FLARE"
            ],
            reference_urls=[
                "https://aclanthology.org/2022.emnlp-main.566",
                "https://arxiv.org/abs/2305.06983"
            ]
        )

    def _normalize_text(self, s: str) -> str:
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _ner(self, s: str) -> List[str]:
        """Extract named entities from the text."""
        doc = self.nlp(s)
        ents = doc.ents
        return [self._normalize_text(e.text) for e in ents]

    def _f1_score(self, pred: str, ref: str) -> float:
        """Compute the f1 score between pred and ref."""
        pred_ents = self._ner(pred)
        ref_ents = self._ner(ref)

        pred_counter = Counter(pred_ents)
        ref_counter = Counter(ref_ents)

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
        pred_answer: str,
        ref_answers: List[str]
    ) -> float:
        """Evaluate the disambig f1 score of an answer."""
        return np.max([self._f1_score(pred_answer, ref_answer) for ref_answer in ref_answers])

    def _compute_batch(
        self,
        pred_answers: List[str],
        ref_answers: List[List[str]]
    ) -> List[float]:
        """Evaluate the disambig f1 score of a batch of answers."""
        scores = []
        for pred, refs in tqdm(zip(pred_answers, ref_answers), desc="Computing Disambig F1", total=len(pred_answers)):
            scores.append(self._compute_one(pred, refs))
        return scores