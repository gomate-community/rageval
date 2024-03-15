from dataclasses import dataclass
from typing import List, Tuple

import datasets
from datasets import Dataset

from rageval.metrics import Metric, add_attribute

_DESCRIPTION = """\
    ChrF and ChrF++ are two MT evaluation metrics. They both use the F-score statistic for character n-gram matches, and ChrF++ adds word n-grams as well which correlates more strongly with direct assessment.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    predictions (list of str): The predicted sentences.
    references (list of list of str): The references. There should be one reference sub-list for each prediction sentence.
    char_order (int): Character n-gram order. Defaults to `6`.
    word_order (int): Word n-gram order. If equals to `2`, the metric is referred to as chrF++. Defaults to `0`.
    beta (int): Determine the importance of recall w.r.t precision. Defaults to `2`.
    lowercase (bool): if `True`, enables case-insensitivity. Defaults to `False`.
    whitespace (bool): If `True`, include whitespaces when extracting character n-grams.
    eps_smoothing (bool): If `True`, applies epsilon smoothing similar

Optional Args:
    None

Functions:
    _validate_data: validate the dataset format.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "The relationship between cats and dogs is not exactly friendly.",
    ...         "a good bookshop is just a genteel black hole that knows how to read."
    ...     ],
    ...     "gt_answers": [
    ...         ["The relationship between dogs and cats is not exactly friendly.", ],
    ...         ["A good bookshop is just a genteel Black Hole that knows how to read."]
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerCHRFCorrectness()
    >>> metric.mtype
    'AnswerCorrectness'
    >>> s, ds = metric.compute(dataset)
    >>> type(ds)
    <class 'datasets.arrow_dataset.Dataset'>
"""

_CITATION = """\
@inproceedings{popovic-2015-chrf,
    title = "chr{F}: character n-gram {F}-score for automatic {MT} evaluation",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Tenth Workshop on Statistical Machine Translation",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W15-3049",
    doi = "10.18653/v1/W15-3049",
    pages = "392--395",
}
@inproceedings{popovic-2017-chrf,
    title = "chr{F}++: words helping character n-grams",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Second Conference on Machine Translation",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-4770",
    doi = "10.18653/v1/W17-4770",
    pages = "612--618",
}
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerCHRFCorrectness(Metric):
    """Estimates the CHRF between answers and ground truth answers."""

    name = "answer_chrf"

    ALIAS = ['answer_chrf']

    def __init__(
            self,
            char_order: int = 6,
            word_order: int = 0,
            beta: int = 2,
            lowercase: bool = False,
            whitespace: bool = False,
            eps_smoothing: bool = False
    ):
        """
        Explicitly initialize AnswerCHRFCorrectness.

        Ensure all parent classes are initialized.
        """
        super().__init__()
        self._required_columns = ['answers', 'gt_answers']
        self.char_order = char_order
        self.word_order = word_order
        self.beta = beta
        self.lowercase = lowercase
        self.whitespace = whitespace
        self.eps_smoothing = eps_smoothing

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
                    "answers": datasets.Value("string"),
                    "gt_answers": datasets.Value("string")
                }
            ),
            codebase_urls=["https://github.com/huggingface/datasets/blob/main/metrics/chrf/chrf.py"],
            reference_urls=["https://aclanthology.org/W15-3049.pdf", "https://aclanthology.org/W17-4770", "https://www.aclweb.org/anthology/W18-6319"]
        )

    def _validate_data(self, dataset: datasets.Dataset) -> bool:
        """Validate the of the input dataset."""
        super()._validate_data(dataset)
        if not all(isinstance(answer, str) for answer in dataset["answers"]):
            raise ValueError("The type of answers should be a string.")
        if not all(isinstance(a, List) or not all(isinstance(item, str) for item in a) for a in dataset["gt_answers"]):
            raise ValueError("The type of gt_answers should be a list of strings.")

    def compute(
        self,
        dataset: Dataset,
        batch_size: int = None,
    ) -> Tuple[float, Dataset]:
        """Evaluate the dataset."""
        chrf = datasets.load_metric("chrf")
        predictions = list(dataset["answers"])
        references = list(dataset["gt_answers"])
        result = chrf.compute(predictions=predictions,
                              references=references,
                              char_order=self.char_order,
                              word_order=self.word_order,
                              beta=self.beta,
                              lowercase=self.lowercase,
                              whitespace=self.whitespace,
                              eps_smoothing=self.eps_smoothing)
        scores = [chrf.compute(predictions=[predictions[i]],
                               references=[references[i]],
                               char_order=self.char_order,
                               word_order=self.word_order,
                               beta=self.beta,
                               lowercase=self.lowercase,
                               whitespace=self.whitespace,
                               eps_smoothing=self.eps_smoothing)['score'] for i in range(len(predictions))]

        return result['score'], dataset.add_column(f"{self.name}", scores)

    def _compute_batch(self, dataset: Dataset) -> list:
        pass
