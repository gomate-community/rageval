from collections import Counter
from nltk import ngrams

import datasets
import evaluate

_DESCRIPTION = """
Distinct 1/2 measures the diversity of generated text by calculating the ratio of unique n-grams to the total number of n-grams.
"""

_KWARGS_DESCRIPTION = """
Args:
    hypothesis (list of str): List of generated texts for which distinct metrics are computed.

Returns:
    tuple: Tuple containing Distinct-1 and Distinct-2 scores.

Examples:
    Example 1:
    >>> hypothesis = ["This is the first sentence.", "This is the second sentence."]
    >>> get_distinct_score(hypothesis)
    (0.8, 0.6)
"""

_CITATION = """
@misc{selfmemory2023,
    title={Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory},
    author={Author Name},
    year={2023},
    eprint={URL},
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

    return distinct_1, distinct_2


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Distinct(evaluate.Metric):
    """Distinct 1/2 metric for text generation."""

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
            reference_urls=None,
        )

    def _compute(
        self,
        hypothesis,
    ):
        """Compute Distinct-1 and Distinct-2 metrics."""
        return get_distinct_score(hypothesis)
