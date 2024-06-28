import numpy as np

from rageval.metrics import Metric, add_attribute
import datasets

_DESCRIPTION = """\
Hit Ratio (HR@K) measures the proportion of users for whom at least one item in their ground-truth set is present in the top-K recommendations.
"""

_KWARGS_DESCRIPTION = """\
Args:
    pos_index (numpy.ndarray): Array representing the position index of items.
Returns:
    dict: Dictionary containing the Hit Ratio metric result.
Examples:
    Example 1:
    >>> pos_index = np.array([[0, 1, 0], [1, 0, 1]])
    >>> hit_metric = Hit(config)
    >>> hit_metric.calculate_metric(pos_index)
    {'hit': array([0, 1])}
"""

_CITATION = """\
@misc{HRMetric2020,
    title={Recommendation System Evaluation Metrics},
    author={Li, Kaiyuan and Feng, Zhichao and Pan, Xingyu and Lin, Zihan},
    year={2020},
    url={https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870}
}
"""


@add_attribute('mtype', 'RecommendationSystem')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class HitRatio(Metric):
    """Hit Ratio (HR@K) metric for recommendation systems."""

    name = "hit_ratio"

    ALIAS = ['hit_ratio', 'hr']

    def __init__(self, config):
        super().__init__()
        self.config = config

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
                    "pos_index": datasets.Array(dtype="int64", shape=[None, None]),
                }
            ),
            reference_urls=["https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870"]
        )

    def calculate_metric(self, pos_index):
        """Calculate Hit Ratio metric."""
        result = np.cumsum(pos_index, axis=1)
        return {'hit': (result > 0).astype(int)}

    def _compute_batch(self, pos_index):
        """Compute Hit Ratio metric for a batch of data."""
        results = []
        for indices in pos_index:
            result = np.cumsum(indices)
            hit = (result > 0).astype(int)
            results.append({'hit': hit})
        return results

    def compute(self, pos_index):
        """Compute Hit Ratio metric."""
        return self._compute_batch(pos_index)
