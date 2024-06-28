import numpy as np

import datasets
from sklearn.metrics import ndcg_score

from rageval.metrics import add_attribute, Metric

_DESCRIPTION = """\
Normalized Discounted Cumulative Gain (NDCG) is a metric used to evaluate the quality of ranked retrieval results.
It takes into account the position of relevant documents in the ranking and assigns higher scores to relevant documents that appear higher in the list.
"""

_KWARGS_DESCRIPTION = """\
Args:
    relevance_score (numpy.ndarray): Array of relevance scores in output order.
    true_relevance (numpy.ndarray): Array of relevance scores in ideal order.
Returns:
    float: NDCG score computed based on the input relevance scores.
Examples:
    Example 1:
    >>> from sklearn.metrics import ndcg_score
    >>> import numpy as np
    >>> true_relevance = np.asarray([[3, 2, 1, 0, 0]])
    >>> relevance_score = np.asarray([[3, 2, 0, 0, 1]])
    >>> ndcg_score(np.asarray([true_relevance]), np.asarray([relevance_score]))
    0.4949747468305837
"""

_CITATION = """\
@misc{SelfMemory2023,
    title={Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory},
    author={Self Memory Contributors},
    year={2023},
    url={https://github.com/Hannibal046/SelfMemory/tree/58d8b611ad51605091c7555c0f32dce6702dadbf}
}
"""


@add_attribute('mtype', 'RankedRetrievalQuality')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NDCGEvaluation(Metric):
    """Normalized Discounted Cumulative Gain (NDCG) metric for ranked retrieval quality."""

    name = "ndcg_evaluation"

    ALIAS = ['ndcg']

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
                    "relevance_score": datasets.Array(dtype="int64", shape=[None]),
                    "true_relevance": datasets.Array(dtype="int64", shape=[None])
                }
            ),
            reference_urls=["https://example.com/self-memory"]
        )

    def compute(self, relevance_score, true_relevance):
        """Compute NDCG score based on relevance scores."""
        return ndcg_score(np.asarray([true_relevance]), np.asarray([relevance_score]))

    def _compute_batch(self, relevance_score, true_relevance):
        pass
