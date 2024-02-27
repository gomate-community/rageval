# -*- coding: utf-8 -*-

from typing import List, Any, Callable

import datasets
import numpy as np
from dataclasses import dataclass

from rageval.metrics import Metric, add_attribute


_DESCRIPTION = """\
AnswerClaimRecall is the metric build based on NLI model.

For details, see the paper: http://arxiv.org/abs/2305.14627.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    batch_size : int, Batch size for openai completion.

Optional Args:
    None

Functions:
    init_model: initialize the model used in evaluation.
    _verify_by_stance: verify whether the stance of args:`claims` can be supported by args:`answer`.
    _compute_one: compute the score by measure whether the args:`claims` can be supported by args:`answers`.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {"answers": ["test answer"], "gt_answers": [["test context"]]}
    >>> dataset = Dataset.from_dict(sample)
    >>> model = rl.models.NLIModel('text-classification', 'hf-internal-testing/tiny-random-RobertaPreLayerNormForSequenceClassification')
    >>> metric = rl.metrics.AnswerClaimRecall()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> metric.init_model(model)
    >>> s, ds = metric.compute(dataset, batch_size=1)
    >>> assert s == 0 or s == 1
    >>> type(ds)
    <class 'datasets.arrow_dataset.Dataset'>
"""

_CITATION = """\
@misc{gao2023enabling,
      title={Enabling Large Language Models to Generate Text with Citations},
      author={Tianyu Gao and Howard Yen and Jiatong Yu and Danqi Chen},
      year={2023},
      eprint={2305.14627},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerClaimRecall(Metric):

    name = "answer_claim_recall"

    def __init__(self):
        """Explicitly initialize the AnswerClaimRecall to ensure all parent class initialized."""
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
                    "answers": datasets.Value("string"),
                    "gt_answers": datasets.Sequence(datasets.Value("string"))
                }
            ),
            codebase_urls=["https://github.com/princeton-nlp/ALCE"],
            reference_urls=["http://arxiv.org/abs/2305.14627"]
        )

    def init_model(self, model: Callable):
        """Initializee the LLM model."""
        self.model = model

    def _verify_by_stance(self, answer: str, claims: List[str]) -> Any:
        """Verify the faithfulness of the `claim` based on `evidences`."""
        labels = []
        for claim in claims:
            label = self.model.infer(premise=answer, hypothesis=claim)
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
        claims: List[str]
    ) -> float:
        """
        Evaluate the correctness of an answer.

        Firstly, split the gt_answer into a set of claims. (There are many ways to obtain claims.)
        It is assumed that claims have been obtained here.
        Then, compute the faithfulness score of each claim. The faithfulness is a binary score.
        Finally, aggregate all faithfulness score of each claim.
        """

        detail_results = []
        scores = []

        for i, claim in enumerate(claims):
            # obtain the faithfulness of each claim by language inference model.
            label = self._verify_by_stance(answer, claims)
            detail_results.append({
                "answer": answer,
                "claim": claim,
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
        Evaluate the correctness of a batch of answers.

        Firstly, split the gt_answer into a set of claims. (There are many ways to obtain claims.)
        It is assumed that claims have been obtained here.
        Then, compute the faithfulness score of each claim. The faithfulness is a binary score.
        Finally, aggregate all faithfulness score of each claim.
        """

        answers, claims = (
            dataset["answers"],
            dataset["gt_answers"],
        )

        results = []
        for i, answer in enumerate(answers):
            r = self._compute_one(answer, claims[i])
            results.append(r)
        return results
