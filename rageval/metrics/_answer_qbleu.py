import json
from typing import List

import datasets
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from rageval.metrics import Metric, add_attribute

_DESCRIPTION = """\
Q-BLEU evaluates response generation quality based on BLEU score for question responses.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str

Optional Args:
    None

Functions:
    _compute_q_bleu: Compute Q-BLEU score between predicted responses and ground truth responses.
    evaluation: Perform Q-BLEU evaluation on experiment results.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "hyp": ["response1", "response2"],
    ...     "ref": [["reference1"], ["reference2"]],
    ...     "context": "context1",
    ...     "da_pred": "prediction1"
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.QBLEU()
    >>> metric.mtype
    'ResponseGeneration'
    >>> score, results = metric.compute(dataset['hyp'], dataset['ref'], 1)
    >>> assert 0 <= score <= 1
"""

_CITATION = """\
"""


@add_attribute('mtype', 'ResponseGeneration')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class QBLEU(Metric):
    """Q-BLEU evaluates response generation quality based on BLEU score for question responses."""

    name = "q_bleu"

    ALIAS = ['q_bleu']

    def __init__(self):
        """Explicitly initialize QBLEU to ensure all parent class initialized."""
        super().__init__()

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
                    "hyp": datasets.Sequence(datasets.Value("string")),
                    "ref": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "context": datasets.Value("string"),
                    "da_pred": datasets.Value("string")
                }
            ),
            codebase_urls=[],
            reference_urls=[]
        )

    def _compute_q_bleu(self, q_hyps: List[str], q_refs: List[List[str]]) -> float:
        """Compute Q-BLEU score between predicted responses and ground truth responses."""
        smoothing_function = SmoothingFunction().method2
        return corpus_bleu(q_refs, q_hyps, smoothing_function=smoothing_function)

    def _compute_batch(
        self,
        pred_responses: List[str],
        ref_responses: List[List[str]],
        **kwargs
    ) -> List[float]:
        """Compute Q-BLEU score for a batch of responses."""
        return [self._compute_q_bleu(q_hyps, q_refs) for q_hyps, q_refs in zip(pred_responses, ref_responses)]

    def evaluation(self, experiment: str):
        """Perform Q-BLEU evaluation on experiment results."""
        jsondata = json.load(open(f'./data/result/result_{experiment}.json'))

        q_hyps, q_refs, _, _ = zip(*[
            (line['hyp'].split(' '), [line['ref'].split(' ')], line['context'], line['da_pred'])
            for line in jsondata
            if line['da_context'][-1] in ['question', '<Question>']
        ])

        print('Q-BLEU: {:.4f}'.format(self._compute_q_bleu(q_hyps, q_refs)))
