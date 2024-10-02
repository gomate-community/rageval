from dataclasses import dataclass
from typing import List, Tuple
import evaluate

import datasets
from rageval.metrics import Metric, add_attribute
from bert_score import BERTScorer
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

_DESCRIPTION = """\
BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language generation tasks.

For details, see the paper: https://openreview.net/forum?id=SkeHuCVFDr
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    lang : str, Language of the text. Default is "en".
    rescale_with_baseline : bool, Whether to rescale the score with pre-computed baseline. Not affect BERTScore's correlation with human judgment. Default is False. For more details, see https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md

Optional Args:
    None

Functions:
    _clean: clean special word in sentence.
    _compute_one: compute bleu score for single prediction with its references
    _compute_batch: compute bleu score for a batch of predictions with their references

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "It is a guide to action which ensures that the military always obeys the commands of the party.",
    ...         "It is to insure the troops forever hearing the activity guidebook that party direct.",
    ...     ],
    ...     "gt_answers": [
    ...         [
    ...             "It is a guide to action that ensures that the military will forever heed Party commands.",
    ...             "It is the guiding principle which guarantees the military forces always being under the command of the Party.",
    ...             "It is the practical guide for the army always to heed the directions of the party.",
    ...         ],
    ...         [
    ...             "It is a guide to action that ensures that the military will forever heed Party commands.",
    ...             "It is the guiding principle which guarantees the military forces always being under the command of the Party.",
    ...             "It is the practical guide for the army always to heed the directions of the party.",
    ...         ]
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.AnswerBERTScore(lang='en', rescale_with_baseline=True)
    >>> metric.mtype
    'AnswerCorrectness'
    >>> score, results = metric.compute(dataset["answers"], dataset["gt_answers"], 1)
    >>> round(score, 2)
    0.55
    >>> round(results[0], 1)
    0.7
"""


_CITATION = """\
@inproceedings{bert-score,
    title={BERTScore: Evaluating Text Generation with BERT},
    author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=SkeHuCVFDr}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerBERTScore(Metric):
    """BERTScore depends on the model and language pair selected."""

    name = "answer_bert_score"

    ALIAS = ['answer_bert_score']

    def __init__(self, lang: str = "en", rescale_with_baseline=False):
        """Explicitly initialize the AnswerBERTScore to ensure all parent class initialized."""
        super().__init__()
        self.scorer = BERTScorer(lang=lang, rescale_with_baseline=rescale_with_baseline)
        self.info = evaluate.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            features=datasets.Features(
                {
                    "answers": datasets.Value("string"),
                    "gt_answers": datasets.Sequence(datasets.Value("string"))
                }
            ),
            codebase_urls=[
                "https://github.com/Tiiiger/bert_score/tree/master",
            ],
            reference_urls=["https://openreview.net/forum?id=SkeHuCVFDr"]
        )

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _compute_one(
        self,
        pred_answers: str,
        ref_answers: List[str]
    ) -> float:
        """Compute the BERTscore for a pair of predictions and references."""
        P, R, F1 = self.scorer.score([pred_answers] * len(ref_answers), ref_answers)
        return F1.max().tolist()
