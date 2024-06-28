import numpy as np
from rageval.metrics import Metric, add_attribute
import datasets

_DESCRIPTION = """\
Mean Reciprocal Rank (MRR) is a metric for evaluating the performance of information retrieval systems.
It measures the average reciprocal rank of relevant items retrieved in the top-k results.
"""

_KWARGS_DESCRIPTION = """\
Args:
    data_samples (list): List of dictionaries containing "query" and "code" keys.
Returns:
    result (dict): Dictionary containing the MRR evaluation score.
Examples:
    >>> data = [
    ...     {"query": "example_query", "code": ["code1", "code2", "code3", "example_query", "code4"]},
    ...     {"query": "another_query", "code": ["code5", "code6", "another_query", "code7", "code8"]}
    ... ]
    >>> metric = rl.metrics.MRREvaluation()
    >>> metric.mtype
    'InformationRetrievalQuality'
    >>> score = metric.compute(data)
    >>> score['eval_mrr']
    0.5
"""
_CITATION = """\
@misc{OpenMatch2022,
    title={Structure-aware language model pretraining improves dense retrieval on structured data},
    author={OpenMatch Contributors},
    year={2022},
    url={https://github.com/OpenMatch/OpenMatch/blob/master/README.md}
}
"""


@add_attribute('mtype', 'InformationRetrievalQuality')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MRREvaluation(Metric):
    """Mean Reciprocal Rank (MRR) metric for information retrieval."""

    name = "mrr_evaluation"

    ALIAS = ['mrr']

    def __init__(self):
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
                    "query": datasets.Value("string"),
                    "code": datasets.Sequence(datasets.Value("string"))
                }
            ),
            reference_urls=["https://github.com/OpenMatch/OpenMatch/blob/master/README.md"]
        )

    def _validate_data(self, data_samples):
        """Validate the input data format."""
        for item in data_samples:
            if "query" not in item or "code" not in item:
                raise ValueError("Each item in data_samples must have 'query' and 'code' keys.")

    def compute(self, data_samples):
        """Compute Mean Reciprocal Rank (MRR) for the input data."""
        self._validate_data(data_samples)
        ranks = []
        for item in data_samples:
            rank = 0
            find = False
            url = item["query"]
            for idx in item["code"][:100]:  # Consider only the first 100 codes
                # MRR@100
                if find is False:
                    rank += 1
                if idx == url:
                    find = True
            if find:
                ranks.append(1 / rank)
            else:
                ranks.append(0)
        result = {"eval_mrr": float(np.mean(ranks))}
        return result

    def _compute_batch(self, data_samples):
        pass
