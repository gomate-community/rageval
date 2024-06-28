from collections import Counter
from dataclasses import dataclass
from typing import List

import datasets
import evaluate
from nltk import ngrams

from rageval.metrics import add_attribute, Metric

_DESCRIPTION = """
Distinct 1/2 measures the diversity of generated text by calculating the ratio of unique n-grams to the total number of n-grams.
"""

_KWARGS_DESCRIPTION = """
Args:
    hypothesis (list of str): List of generated texts for which distinct metrics are computed.

Returns:
    tuple: Tuple containing Distinct-1 and Distinct-2 scores.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "hypothesis": [
    ...         "This is the first sentence.",
    ...         "This is the second sentence."
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.Distinct()
    >>> metric.mtype
    'Diversity'
    >>> scores = metric.compute(dataset['hypothesis'])
    >>> scores['distinct_1']
    0.8
    >>> scores['distinct_2']
    0.6
"""

_CITATION = """
@misc{selfmemory2023,
    title={Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory},
    author={Xin Cheng and Di Luo and Xiuying Chen and Lemao Liu and Dongyan Zhao and Rui Yan},
    year={2023},
    eprint={2305.02437},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


def get_distinct_score(hypothesis):
    """Compute Distinct-1 and Distinct-2 metrics."""
    unigram_counter = Counter()
    bigram_counter = Counter()

    for hypo in hypothesis:
        tokens = hypo.split()
        unigram_counter.update(tokens)
        bigram_counter.update(ngrams(tokens, 2))

    distinct_1 = len(unigram_counter) / sum(unigram_counter.values())
    distinct_2 = len(bigram_counter) / sum(bigram_counter.values())

    return {"distinct_1": distinct_1, "distinct_2": distinct_2}


@dataclass
@add_attribute('mtype', 'TextDiversity')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerDistinct12(Metric):
    """Distinct 1/2 metric for text generation."""

    name = "distinct"

    ALIAS = ['distinct']

    def __init__(self):
        """
        Explicitly initialize Distinct metric.
        Ensure all parent classes are initialized.
        """
        super().__init__()

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"
    
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "hypothesis": datasets.Sequence(datasets.Value("string", description="List of generated texts for which distinct metrics are computed.")),
                }
            ),
            codebase_urls=["https://github.com/Hannibal046/SelfMemory/tree/58d8b611ad51605091c7555c0f32dce6702dadbf"],
            reference_urls=["https://arxiv.org/abs/2305.02437"],
        )

    def _compute(
        self,
        hypothesis: List[str],
    ) -> dict:
        """Compute Distinct-1 and Distinct-2 metrics."""
        return get_distinct_score(hypothesis)
