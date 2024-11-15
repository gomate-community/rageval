from dataclasses import dataclass
from typing import Optional, Iterable
from refchecker.extractor import LLMExtractor
from refchecker.checker import LLMChecker

import evaluate
import datasets
import os
from rageval.metrics import Metric, add_attribute
import numpy as np


_DESCRIPTION = """\
ClaimFaithfulness is a metric that evaluates to what extend does the answer follows the given evidences. 

It is calculated by first utilizing the open-source tool RefChecker to extract claims from the generated text, and then use the same tool to check whether evidences can entail each claim. The ultimate measure is the total number of entailment, providing insight into the faithfulness to given evidences in the model's outputs.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str

Optional Args:
    None

Functions:
    _compute_one: Evaluating the faithfulness of claims generated.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "A",
    ...         "C",
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = ClaimFaithfulness(model = "openai//home/gomall/models/Qwen2-7B-Instruct", api_base = "http://project.gomall.ac.cn:30590/notebook/tensorboard/wangwenshan/1161/v1"， api_key = "sk-123456789")
    >>> metric.mtype
    'answer_informativeness'
"""

@dataclass
@add_attribute('mtype', 'answer_informativeness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ClaimFaithfulness(Metric):
    """Estimates the faithfulness of claims contained in answers."""

    name = "claim_faithfulness"

    ALIAS = ['claim_faithfulness']

    def __init__(self, model: str = "openai//home/gomall/models/Qwen2-7B-Instruct",
                 api_base: str = "http://localhost:5000/v1",
                 api_key: str = "sk-123456789"):
        """
        Explicitly initialize ClaimFaithfulness.

        Ensure all parent classes are initialized.
        """
        self.extractor = LLMExtractor(model=model, batch_size=8, api_base=api_base)
        self.checker = LLMChecker(model=model, batch_size=8, api_base=api_base)
        os.environ['OPENAI_API_KEY'] = api_key
        super().__init__()
        self.info = evaluate.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation="",
            homepage="",
            features=datasets.Features(
                {
                    "answers": datasets.Value("string"),
                }
            ),
            codebase_urls=[],
            reference_urls=[]
        )

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"  # pragma: no cover

    def _compute_one(
        self,
        answer: str,
        question: str,
        context: str,
        *args: Optional[Iterable],
    ) -> float:
        """Evaluating the richness of claims contained in answers."""
        extraction_results = self.extractor.extract(
            batch_responses=[answer],
            batch_questions=[question],
            max_new_tokens=1000
        )
        claims = [[c.content for c in res.claims] for  res in extraction_results]
        merge_psg = False
        checking_results = self.checker.check(
                            batch_claims=claims,
                            batch_references=[context],
                            batch_questions=[question],
                            max_reference_segment_length=0,
                            merge_psg=merge_psg,
                            is_joint=True,
                            joint_check_num=5,
                            sagemaker_client=None,
                            sagemaker_params=None,
                            sagemaker_get_response_func=None,
                        )
        def to_bool(checking_results):
            if isinstance(checking_results, str):
                return checking_results == "Entailment"
            return np.array([to_bool(res) for res in checking_results])
        
        retrieved2response = to_bool(checking_results)
        faithful = np.max(retrieved2response, axis=2)
        faithfulness_score = np.mean(faithful)

        return faithfulness_score