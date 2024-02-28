# -*- coding: utf-8 -*-

from typing import List, Any, Callable, Union
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
    init_model: initialize the `rouge_type` used in evaluation.
    _compute_one: compute the score by measure whether the args:`answer` contains short answer in list:`gt_answers`.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {"answers": ["test answer"],"gt_answers":[["test gt_answer", "test groundtruth answer"]]}
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerRougeCorrectness()
    >>> metric.init_model('rougeL')
    >>> score, results = metric.compute(dataset, batch_size= 1)
    >>> assert 0 <= score <= 1
    >>> type(results)
    <class 'datasets.arrow_dataset.Dataset'>
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
"""

@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerRougeCorrectness(Metric):

    name = "answer_rouge_correctness"

    def __init__(self):
        """Explicitly initialize the AnswerRougeCorrectness to ensure all parent class initialized."""
        self._required_columns = ['answers', 'gt_answers']
        super().__init__()

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

    def init_model(self, rouge_type: str, tokenizer: Union[Callable, None] = None):
        """Initialize the rouge type."""
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True, tokenizer=tokenizer)

    def _compute_one(self, answer: str, gt_answers: List[str]) -> float:
        """Evaluate the ROUGE between a single answer and groundtruth answers."""
        score = self.scorer.score_multi(gt_answers, answer)
        return score[self.rouge_type].fmeasure

    def _compute_batch(self, dataset: Dataset) -> list:
        """Evaluate the ROUGE of a batch of answers."""
        results = [self._compute_one(answer, gt_answer) for answer, gt_answer in zip(dataset["answers"], dataset["gt_answers"])]
        return results
