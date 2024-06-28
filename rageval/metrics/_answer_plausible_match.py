from dataclasses import dataclass
from typing import List

import datasets
import numpy as np

from rageval.metrics import add_attribute, Metric

_DESCRIPTION = """\
Plausible Match (PM) measures the retrieval performance based on similarity thresholds between retrieved and ground-truth items.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    modal : str, Specifies whether the evaluation is based on 'image' or 'text' modalities.
    pdists : numpy.ndarray, Pairwise distance matrix for similarity calculations.
    iid_to_idx : dict, Mapping from item IDs to indices in the distance matrix.
    val_image_ids : list, List of valid image IDs for evaluation.
    max_thres : int, Maximum threshold for similarity.

Returns:
    dict: Dictionary containing RP (Recall Precision) and R@1 (Recall at 1) metrics per threshold.

Examples:
    Example 1:
    >>> data = {}  # Your data dictionary
    >>> modal = 'image'
    >>> pdists = np.zeros((10, 10))  # Replace with your pairwise distance matrix
    >>> iid_to_idx = {}  # Replace with your mapping
    >>> val_image_ids = []  # Replace with your list of valid image IDs
    >>> max_thres = 3
    >>> metric = PlausibleMatch()
    >>> rp_per_query, r1_per_query = metric.evaluate(data, modal, pdists, iid_to_idx, val_image_ids, max_thres)
"""

_CITATION = """\
@misc{PCME2021,
    author={NAVER Corp.},
    title={Plausible Match Evaluation for Retrieval Performance},
    year={2021},
    url={https://example.com/pcme}
}
"""


@dataclass
@add_attribute('mtype', 'RetrievalPerformance')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PlausibleMatch(Metric):
    """Plausible Match (PM) metric for retrieval performance."""

    name = "plausible_match"

    ALIAS = ['plausible_match', 'pm']

    def __init__(self):
        """
        Explicitly initialize PlausibleMatch.

        Ensure all parent classes are initialized.
        """
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
                    "data": datasets.Value("dict"),
                    "modal": datasets.Value("str"),
                    "pdists": datasets.Array(dtype="float32", shape=[None, None]),
                    "iid_to_idx": datasets.Value("dict"),
                    "val_image_ids": datasets.Sequence(datasets.Value("str")),
                    "max_thres": datasets.Value("int32")
                }
            ),
            codebase_urls=[],
            reference_urls=["https://example.com/pcme"]
        )

    def evaluate(
        self,
        data,
        modal: str,
        pdists: np.ndarray,
        iid_to_idx: dict,
        val_image_ids: List[str],
        max_thres: int = 3
    ) -> tuple:
        """Evaluate Plausible Match metrics."""
        rp_per_query_per_amb = {}
        r1_per_query_per_amb = {}

        for thres in range(max_thres):
            rp_per_query = []
            r1_per_query = []
            for key, _data in data.items():
                if modal == 'image':
                    iid = key
                elif modal == 'text':
                    iid = _data['query']['image_id']
                if iid not in val_image_ids:
                    continue
                image_id_key = 'image_id' if modal == 'image' else 'id'
                distance = pdists[iid_to_idx[iid]]
                retrieved_sim = np.array([distance[iid_to_idx[_d[image_id_key]]]
                                          for _d in _data['retrieved']
                                          if _d[image_id_key] in val_image_ids])

                matched = retrieved_sim <= thres
                R = np.sum(matched)
                rp = np.sum(matched[:R]) / R if R > 0 else 0
                rp_per_query.append(rp)
                r1_per_query.append(int(matched[0]) if len(matched) > 0 else 0)
            rp_per_query_per_amb[thres] = rp_per_query
            r1_per_query_per_amb[thres] = r1_per_query

        return rp_per_query_per_amb, r1_per_query_per_amb
