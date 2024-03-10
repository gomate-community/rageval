from dataclasses import dataclass
from typing import List, Callable

import datasets
import numpy as np

from rageval.metrics import Metric, add_attribute
from rageval.utils.check_utils import text_to_sents

_DESCRIPTION = """\
The AnswerNLICorrectness is to measure the correctness of long-form answers. In the original paper, the author first \
use Instruct-GPT(text-davinci-003) to generate three "sub-claims" (based on gold answers) and use a state-of-the-art \
natural-language inference (NLI) model TRUE(Honovich et al., 2022) to check whether the model output entails the \
sub-claims (claim recall).

For details, see the paper: https://arxiv.org/abs/2305.14627.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    batch_size : int, Batch size for openai completion.

Optional Args:
    None

Functions:
    _verify_by_stance: verify whether the stance of args:`claims` can be supported by args:`answer`.
    _compute_one: compute the score by measure whether the args:`claims` can be supported by args:`answers`.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "They went a while before introducing ads, so they could make money, as they needed to  establish "
    ...         "their brand and amass users. Once you have dedicated users, introducing ads won't deter most, but if "
    ...         "you are still new, having ads will deter a lot. The same goes for Uber, it's not that they aren't "
    ...         "making money, it's that they are reinvesting a ton of it to make their service better."
    ...     ],
    ...     "gt_answers": [
    ...         [
    ...             "Firms like Snapchat and Uber need to establish their brand and amass users before introducing "
    ...             "ads.",
    ...             "Introducing ads too early can deter potential users.",
    ...             "Uber is reinvesting a lot of money to make their service better."
    ...         ]
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> nli_model = rl.models.NLIModel(
    ...     'text2text-generation',
    ...     'hf-internal-testing/tiny-random-T5ForConditionalGeneration'
    ... )
    >>> metric = rl.metrics.AnswerNLICorrectness(nli_model=nli_model, decompose_model="nltk")
    >>> metric.mtype
    'AnswerCorrectness'
    >>> s, ds = metric.compute(dataset, batch_size=1)
    >>> assert s == 0 or s == 1
    >>> type(ds)
    <class 'datasets.arrow_dataset.Dataset'>
"""

_CITATION = """\
@misc{gao2023enabling,
      title={Enabling Large Language Models to Generate Text with Citations},
      author={Tianyu Gao and Howard Yen and Jiatong Yu and Danqi Chen},
      year={2023},
      eprint={2305.14627},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerNLICorrectness(Metric):
    """Estimates the correctness of long-form answers based on the NLI model."""

    name = "answer_claim_recall"

    ALIAS = ['answer_claim_recall']

    def __init__(self, nli_model: Callable, decompose_model: str = "gpt-3.5-turbo"):
        """
        Explicitly initialize AnswerNLICorrectness.

        Ensure all parent classes are initialized.
        Ensure nli_model and decompose_model is initialized.
        """
        super().__init__()
        self._required_columns = ['answers', 'gt_answers']
        self.nli_model = nli_model
        self.decompose_model = decompose_model

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            features=datasets.Features(
                {
                    "answers": datasets.Value("string"),
                    "gt_answers": datasets.Value("string")
                }
            ),
            codebase_urls=["https://github.com/princeton-nlp/ALCE"],
            reference_urls=["https://arxiv.org/abs/2305.14627"]
        )

    def _compute_one(
        self,
        answer: str,
        claims: List[str]
    ) -> float:
        """
        Evaluate the correctness of an answer.

        Firstly, split the gt_answer into a set of claims.
        Then, compute the faithfulness score of each claim. The faithfulness is a binary score.
        Finally, aggregate all faithfulness score of each claim.
        """

        detail_results = []
        scores = []

        for i, claim in enumerate(claims):
            # obtain the faithfulness of each claim by language inference model.
            label = self.nli_model.generate_infer(premise=answer, hypothesis=claim)
            detail_results.append({
                "answer": answer,
                "claim": claim,
                "reasoning": "",
                "error": "",
                "factuality": label,
            })
            scores.append(label)
        # Note that the detail_results can be recorded by logger.info
        return np.average(scores)

    def _compute_batch(
        self,
        dataset: datasets.Dataset
    ) -> list:
        """
        Evaluate the correctness of a batch of answers.

        Firstly, split the gt_answer into a set of claims.
        Then, compute the faithfulness score of each claim. The faithfulness is a binary score.
        Finally, aggregate all faithfulness score of each claim.
        """

        if isinstance(dataset["gt_answers"], list):
            if isinstance(dataset["gt_answers"][0], list):
                # gt_answers has been decomposed into claims list
                claims = dataset["gt_answers"]
            elif isinstance(dataset["gt_answers"][0], str):
                # use decompose_model to decompose the gt_answers into claims list
                claims = [text_to_sents(gt_answer, self.decompose_model) for gt_answer in dataset["gt_answers"]]
            else:
                raise ValueError("The type of gt_answers element should be list or string.")
        else:
            raise ValueError("The type of gt_answers should be list.")

        answers = dataset["answers"]

        results = []
        from tqdm import tqdm
        for i, answer in tqdm(enumerate(answers)):
            r = self._compute_one(answer, claims[i])
            results.append(r)
        return results
