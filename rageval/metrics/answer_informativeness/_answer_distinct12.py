from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple
import datasets
from nltk import ngrams
from rageval.metrics import Metric, add_attribute

_DESCRIPTION = """\
Distinct 1/2 measures the diversity of generated text by calculating the ratio of unique n-grams to the total number of n-grams.
"""

_KWARGS_DESCRIPTION = """\
Args:
    pred_answers (list of str): List of generated texts for which distinct metrics are computed.
    n_grams (int): The n-gram order for which distinct metrics are computed.

Returns:
    dict: Dictionary containing Distinct-1 and Distinct-2 scores.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "This is the first sentence.",
    ...         "This is the second sentence."
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerDistinct(1)
    >>> metric.mtype
    'AnswerInformativeness'
    >>> score, results = metric.compute(dataset['answers'])
    >>> score
    0.6
"""

_CITATION = """\
@misc{selfmemory2023,
    title={Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory},
    author={Xin Cheng and Di Luo and Xiuying Chen and Lemao Liu and Dongyan Zhao and Rui Yan},
    year={2023},
    eprint={2305.02437},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


def get_distinct_score(pred_answers: List[str], n_grams:int) -> dict:
    """Compute Distinct-1 and Distinct-2 metrics."""
    c = Counter()
    for answer in pred_answers:
        tokens = answer.split()
        c.update(ngrams(tokens, n_grams))
    
    distinct = len(c) / sum(c.values())
    return distinct

@dataclass
@add_attribute('mtype', 'AnswerInformativeness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerDistinct(Metric):
    """Distinct 1/2 metric for text generation."""

    name = "answer_distinct"

    ALIAS = ['answer_distinct']

    def __init__(self, n_grams: int = 1):
        """
        Explicitly initialize Distinct.

        Ensure all parent classes are initialized.
        """
        super().__init__()
        self.n_grams = n_grams

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
                    "pred_answers": datasets.Value("string"),
                }
            ),
            codebase_urls=["https://github.com/Hannibal046/SelfMemory/blob/58d8b611ad51605091c7555c0f32dce6702dadbf/src/utils/metrics_utils.py"],
            reference_urls=["https://arxiv.org/abs/2305.02437"]
        )

    def _validate_data(
        self, 
        pred_answers: Optional[Iterable] = None,
        ref_answers: Optional[Iterable] = None,
    ) -> bool:
        """Validate the input data."""
        assert isinstance(pred_answers, str) or isinstance(pred_answers, list)

    def compute(
        self, 
        pred_answers: Optional[Iterable] = None, 
    ) -> Tuple[float, List[float]]:
        return get_distinct_score(pred_answers, self.n_grams), [get_distinct_score([pred_answer], self.n_grams) for pred_answer in pred_answers]
