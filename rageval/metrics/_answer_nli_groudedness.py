# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
from abc import ABC
from typing import List, Any, Callable


import datasets
# from datasets import Dataset
from dataclasses import dataclass

from rageval.metrics import Metric, add_attribute
from rageval.utils import text_to_sents


_DESCRIPTION = """\
AnswerNLIGroundedness is the metric build based on NLI model.

For details, see the paper: https://arxiv.org/abs/XXX.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    batch_size : int, Batch size for openai completion.

Optional Args:
    None

Functions:
    init_model: initialize the model used in evaluation.
    _verify_by_stance: verify whether the stance of args:`claim` can be supported by args:`evidences`.
    _compute_one: compute the score by measure whether the args:`answer` can be supported by args:`evidences`.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {"questions": ["this is a test"],"answers": ["test answer"],"contexts": [["test context"]]}
    >>> dataset = Dataset.from_dict(sample)
    >>> model = rl.models.NLIModel('text-classification', 'hf-internal-testing/tiny-random-RobertaPreLayerNormForSequenceClassification')
    >>> metric = rl.metrics.AnswerNLIGroundedness()
    >>> metric.mtype
    'AnswerGroundedness'
    >>> metric.init_model(model)
    >>> s,ds = metric.compute(dataset, batch_size=1)
    >>> assert s == 0 or s == 1
    >>> type(ds)
    <class 'datasets.arrow_dataset.Dataset'>
"""

_CITATION = """\
@inproceeding={
    title={},
    author={},
    booklet={},
    year={2021}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerGroundedness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerNLIGroundedness(Metric):

    name = "answer_nli_groundedness"

    def __init__(self):
        """Explicitly initialize the AnswerNLIGroundedness to ensure all parent class initialized."""
        self._required_columns = ['answers', 'contexts']
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

    def init_model(self, model: Callable):
        """Initializee the LLM model."""
        self.model = model

    def _verify_by_stance(self, claim: str, evidences: List[str]) -> Any:
        """Verify the faithfulness of the `claim` based on `evidences`."""
        labels = []
        for evidence in evidences:
            label = self.model.infer(premise=evidence, hypothesis=claim)
            labels.append(label)
        if "support" in labels:
            return True
        elif "refute" in labels:
            return False
        else:
            return False

    def _compute_one(
        self,
        answer: str,
        evidences: List[str]
    ) -> float:
        """
        Evaluate the groundedness of an answer.

        Firstly,split the answer into a set of claims.
        Then, compute the faithfulness score of each claim. The faithfulness is a binary score.
        Finally, aggregate all faithfulness score of each claim.
        """

        detail_results = []
        # decompose answers into a list of claim
        claims = text_to_sents(answer)
        scores = []

        for i, claim in enumerate(claims):
            # obtain the faithfulness of each claim by language inference model.
            label = self._verify_by_stance(claim, evidences)
            detail_results.append({
                "claim": claim,
                "evidence": evidences,
                "reasoning": "",
                "error": "",
                "factuality": label,
            })
            scores.append(label)
        # Note that the detail_results can be recorded by logger.info
        return np.average(scores)

    def _compute_batch(
        self,
        dataset: datasets.Dataset
    ) -> list:
        """
        Evaluate the groundedness of a batch of answers.

        Firstly,split the answer into a set of claims.
        Then, compute the faithfulness score of each claim. The faithfulness is a binary score.
        Finally, aggregate all faithfulness score of each claim.
        """

        answers, contexts = (
            dataset["answers"],
            dataset["contexts"],
        )

        results = []
        for i, answer in enumerate(answers):
            # decompose answers into a list of claim
            r = self._compute_one(answer, contexts[i])
            results.append(r)
        return results
