from collections import Counter
from typing import List, Union

import datasets

import evaluate

_DESCRIPTION = """
Named entity-based F1 score evaluates the precision, recall, and F1 score based on named entities in text predictions compared to ground truth.
"""

_KWARGS_DESCRIPTION = """
Args:
    prediction (str): The predicted text containing named entities.
    ground_truth (Union[str, List[str]]): The ground truth text or list of texts containing named entities.
    ground_truth_id (str, optional): Identifier for the ground truth. Defaults to None.
    debug (bool, optional): Whether to print debug information. Defaults to False.

Returns:
    dict: Dictionary containing entity F1 score, precision, recall, and average number of entities per ground truth text.

Examples:
    Example 1:
    >>> prediction = "Apple is a company founded by Steve Jobs."
    >>> ground_truth = ["Apple Inc. was founded by Steve Wozniak and Steve Jobs."]
    >>> entity_f1_score(prediction, ground_truth)
    {'ent_f1': 0.5, 'ent_precision': 0.5, 'ent_recall': 0.5, 'num_ent': 7.0}
"""

_CITATION = """
@article{jiang2023flare,
    title={Active Retrieval Augmented Generation},
    author={Zhengbao Jiang and Frank F. Xu and Luyu Gao and Zhiqing Sun and Qian Liu and Jane Dwivedi-Yu and Yiming Yang and Jamie Callan and Graham Neubig},
    year={2023},
    eprint={2305.06983},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


def entity_f1_score(
    cls,
    prediction: str,
    ground_truth: Union[str, List[str]],
    ground_truth_id: str = None,
    debug: bool = False,
):
    """Calculate entity F1 score, precision, recall, and average number of entities per ground truth text."""
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    p = r = f1 = num_ent = 0

    for gold in ground_truth:
        pred_ents = [cls.normalize_answer(ent.text) for ent in cls.get_ner(prediction)]
        gold_ents = [cls.normalize_answer(ent.text) for ent in cls.get_ner(gold)]

        common_ents = Counter(pred_ents) & Counter(gold_ents)
        num_common_ents = sum(common_ents.values())

        if debug:
            print('PP', prediction)
            print('GG', gold)
            print('P', pred_ents)
            print('G', gold_ents)
            print('C', common_ents)

        _p = (num_common_ents / len(pred_ents)) if len(pred_ents) else 1
        _r = (num_common_ents / len(gold_ents)) if len(gold_ents) else 1
        _f1 = (2 * _p * _r) / ((_p + _r) or 1)

        p, r, f1 = max(p, _p), max(r, _r), max(f1, _f1)
        num_ent += len(gold_ents)

    num_ent /= len(ground_truth)

    return {'ent_f1': f1, 'ent_precision': p, 'ent_recall': r, 'num_ent': num_ent}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NamedEntityF1(evaluate.Metric):
    """Named entity-based F1 score metric."""

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "prediction": datasets.Value("string", description="The predicted text containing named entities."),
                    "ground_truth": datasets.Value("string", description="The ground truth text or list of texts containing named entities."),
                    "ground_truth_id": datasets.Value("string", description="Identifier for the ground truth."),
                    "debug": datasets.Value("bool", description="Whether to print debug information."),
                }
            ),
            reference_urls=None,
        )

    def _compute(
        self,
        prediction,
        ground_truth,
        ground_truth_id=None,
        debug=False,
    ):
        """Compute named entity-based F1 score."""
        return entity_f1_score(self, prediction, ground_truth, ground_truth_id, debug)
