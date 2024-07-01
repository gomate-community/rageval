# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Callable, Optional

import datasets
from datasets import Dataset
from rouge_score import rouge_scorer

from rageval.metrics import Metric, add_attribute

_DESCRIPTION = """Estimates ROUGE score by estimating answer and groundtruth answers.

ROUGE is case insensitive, so the input text is converted to lower case before computing the score. This metrics is a wrapper around the https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py

"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    rouge_type : str, the rouge type to calculate. Defaults to 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'
        "rouge1": unigram (1-gram) based scoring
        "rouge2": bigram (2-gram) based scoring
        "rougeL": Longest common subsequence based scoring.
        "rougeLSum": splits text using "\n".

Optional Args:
    tokenizer : Callable, a tokenizer can be passed to the scorer, replacing the default tokenizer which tokenizes on whitespace, especially for non-latin languages. For example, the `jieba.cut` can be used for Chinese.

Functions:
    _compute_one: compute the score by measure whether the args:`answer` contains short answer in list:`gt_answers`.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...    "answers": [
    ...        "Some nanomaterials may give rise to various kinds of lung damage."
    ...    ],
    ...    "gt_answers":[
    ...        [
    ...            "Nanoparticles can penetrate the body, affecting the lungs, brain, and other organs,\
    ...             leading to possible respiratory, cardiovascular, and brain health problems.",
    ...            "Due to their small size, nanoparticles can infiltrate the body and impact vital organs,\
    ...             posing risks to respiratory, heart, and neurological health."
    ...        ]
    ...    ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerRougeCorrectness('rougeL')
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)
    >>> assert 0 <= score <= 1
"""

_CITATION = """\
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W04-1013",
    pages = "74--81",
}
@article{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9459--9474},
  year={2020}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerRougeCorrectness(Metric):

    name = "answer_rouge_correctness"

    ALIAS = ['answer_rouge_correctness']

    def __init__(self, rouge_type: str, tokenizer: Optional[Callable] = None):
        """Explicitly initialize the AnswerRougeCorrectness to ensure all parent class initialized as well as initialize the rouge type and tokenizer."""
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True, tokenizer=tokenizer)
        super().__init__()

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            features=datasets.Features(
                {
                    "answers": datasets.Value("string", id="sequence"),
                    "gt_answers": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/mim-solutions/rouge_score"],
            reference_urls=[
                "https://aclanthology.org/W04-1013/",
                "https://arxiv.org/abs/2005.11401"
            ]
        )

    def _compute_one(self, pred_answer: str, ref_answers: List[str]) -> float:
        """Evaluate the ROUGE between a single answer and groundtruth answers."""
        score = self.scorer.score_multi(ref_answers, pred_answer)
        return score[self.rouge_type].fmeasure

