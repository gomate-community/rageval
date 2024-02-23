# -*- coding: utf-8 -*-

from typing import List, Any, Callable, Union
from rouge_score import rouge_scorer

from datasets import Dataset
from dataclasses import dataclass

from rageval.metrics import Metric


@dataclass
class AnswerRouge(Metric):
    """
    Estimates ROUGE score by estimating answer and groundtruth answers.

    ROUGE is case insensitive, so the input text is converted to lower case before computing the score.This metrics is a wrapper around the https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py

    Attributes
    ----------
    name : str
    rouge_type : str, the rouge type to calculate. Defaults to 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'
        "rouge1": unigram (1-gram) based scoring
        "rouge2": bigram (2-gram) based scoring
        "rougeL": Longest common subsequence based scoring.
        "rougeLSum": splits text using "\n"
    tokenizer : Callable, optional, for non-latin languages, a tokenizer can be passed to the scorer. For example, the `jieba.cut` can be used for Chinese.

    """

    name = "answer_rouge"

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
