# -*- coding: utf-8 -*-

from typing import List, Any, Callable, Optional
from rouge_score import rouge_scorer
import datasets

from datasets import Dataset
from dataclasses import dataclass

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
    >>> score, results = metric.compute(dataset, batch_size= 1)
    >>> assert 0 <= score <= 1
    >>> type(results)
    <class 'datasets.arrow_dataset.Dataset'>
"""

_CITATION = """\
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
        self._required_columns = ['answers', 'gt_answers']
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
                    "contexts": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[],
            reference_urls=[]
        )

    def _validate_data(self, dataset: Dataset) -> bool:
        super()._validate_data(dataset)
        if not all(isinstance(answer, str) for answer in dataset["answers"]):
            raise ValueError("The type of answers should be a string.")
        if not all(isinstance(a, List) or not all(isinstance(item, str) for item in a) for a in dataset["gt_answers"]):
            raise ValueError("The type of gt_answers should be a list of strings.")

    def _compute_one(self, answer: str, gt_answers: List[str]) -> float:
        """Evaluate the ROUGE between a single answer and groundtruth answers."""
        score = self.scorer.score_multi(gt_answers, answer)
        return score[self.rouge_type].fmeasure

    def _compute_batch(self, dataset: Dataset) -> list:
        """Evaluate the ROUGE of a batch of answers."""
        results = [self._compute_one(answer, gt_answer) for answer, gt_answer in zip(dataset["answers"], dataset["gt_answers"])]
        return results
