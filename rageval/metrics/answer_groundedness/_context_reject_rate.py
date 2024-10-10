from dataclasses import dataclass
from typing import Callable, List, Tuple

import datasets
import numpy as np
import evaluate

from langchain.schema import LLMResult
from tqdm import tqdm

from rageval.metrics import MetricWithLLM, add_attribute
from rageval.utils.prompt import REJECT_RATE_PROMPT

_DESCRIPTION = """\
ContextRejectRate is the metric to measure the unknown robustness of LLM based on the given context.

For details, see the paper: https://arxiv.org/abs/2311.09210.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    batch_size : int, Batch size for openai completion.
    model : Callable, The LLM model to use.

Optional Args:
    None

Functions:
    parse_llm_result: parse the results of LLM
    _compute_batch: compute the score by measure how many rejected answers in all answers.

Examples:
    >>> from datasets import Dataset
    >>> from langchain.llms.fake import FakeListLLM
    >>> import rageval as rl
    >>> sample = {
    ...     "questions": [
    ...         "Why did Bushnell set himself on fire?",
    ...         "Did Bushnell have a wife?"
    ...     ],
    ...     "contexts": [
    ...         [
    ...             ["An active-duty member of the U.S. Air Force has died after he set himself ablaze outside the "
    ...              "Israeli Embassy in Washington, D.C., while declaring that he “will no longer be complicit in "
    ...              "genocide.”"],
    ...             ["The 25-year-old airman, Aaron Bushnell, of San Antonio, Texas, died from his injuries, the "
    ...              "Metropolitan Police Department said Monday."],
    ...             ["Bushnell had walked up to the embassy shortly before 1 p.m. Sunday and began livestreaming on "
    ...              "the video streaming platform Twitch, a person familiar with the matter told The Associated "
    ...              "Press. Law enforcement officials believe he set his phone down and then doused himself in "
    ...              "accelerant and ignited the flames. At one point, he said he “will no longer be complicit in "
    ...              "genocide,” the person said. The video was later removed from the platform, but law enforcement "
    ...              "officials have obtained and reviewed a copy."]
    ...         ],
    ...         [
    ...             ["An active-duty member of the U.S. Air Force has died after he set himself ablaze outside the "
    ...              "Israeli Embassy in Washington, D.C., while declaring that he “will no longer be complicit in "
    ...              "genocide.”"],
    ...             ["The 25-year-old airman, Aaron Bushnell, of San Antonio, Texas, died from his injuries, the "
    ...              "Metropolitan Police Department said Monday."],
    ...             ["Bushnell had walked up to the embassy shortly before 1 p.m. Sunday and began livestreaming on "
    ...              "the video streaming platform Twitch, a person familiar with the matter told The Associated "
    ...              "Press. Law enforcement officials believe he set his phone down and then doused himself in "
    ...              "accelerant and ignited the flames. At one point, he said he “will no longer be complicit in "
    ...              "genocide,” the person said. The video was later removed from the platform, but law enforcement "
    ...              "officials have obtained and reviewed a copy."]
    ...         ],
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> model = FakeListLLM(
    ...     responses=[
    ...         "Answer: An active-duty member of the U.S. Air Force has died after he set himself ablaze outside the "
    ...         "Israeli Embassy in Washington, D.C., while declaring that he “will no longer be complicit in "
    ...         "genocide.”",
    ...         "Answer: sorry, cannot answer the question"
    ...     ]
    ... )
    >>> metric = rl.metrics.ContextRejectRate(model)
    >>> metric.mtype
    'AnswerGroundedness'
    >>> score, results = metric.compute(dataset['questions'], dataset['contexts'], 1)
    >>> assert 0 <= score <= 1
"""

_CITATION = """\
@misc{yu2023chainofnote,
      title={Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models},
      author={Wenhao Yu and Hongming Zhang and Xiaoman Pan and Kaixin Ma and Hongwei Wang and Dong Yu},
      year={2023},
      eprint={2311.09210},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerGroundedness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ContextRejectRate(MetricWithLLM):
    """Estimates context reject rate by measuring how many rejected answers in all answers."""

    name = "context_reject_rate"

    ALIAS = ['context_reject_rate']

    def __init__(self, model: Callable):
        """Explicitly initialize the ContextRejectRate to ensure all parent class initialized."""
        super().__init__(model)
        self.info = evaluate.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            features=datasets.Features(
                {
                    "questions": datasets.Value("string"),
                    "contexts": datasets.Sequence(datasets.Value("string"))
                }
            ),
            codebase_urls=[],
            reference_urls=["https://arxiv.org/abs/2311.09210"]
        )

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"  # pragma: no cover

    def parse_llm_result(self, prompts: List[str], result: LLMResult):
        """Parse the results of LLM based on whether the answer contains the content specified by prompt."""
        responses = [[i.text for i in r] for r in result.generations]
        scores = []
        # for each question-answer pair
        for response in responses:
            answer = response[0]
            if "sorry, cannot answer the question" in answer:
                scores.append(1.)
            else:
                scores.append(0.)
        return scores

    def compute(
        self,
        questions: List[str],
        contexts: List[List[str]],
        batch_size: int,
    ) -> Tuple[float, List[float]]:
        """Evaluate the dataset."""
        scores = []
        length = len(questions)
        for start in tqdm(range(0, length, batch_size)):
            end = start + batch_size
            end = end if end < length else length
            score = self._compute_batch(
                questions[start:end],
                contexts[start:end]
            )
            scores.extend(score)

        return np.average(scores), scores

    def _compute_batch(
        self,
        questions: List[str],
        contexts: List[List[str]],
    ) -> List[float]:
        """Compute the score by measure how many rejected answers in all answers."""

        prompts = []
        for question, context in zip(questions, contexts):
            prompt = REJECT_RATE_PROMPT.format(
                question=question,
                evidence=context
            )
            prompts.append(prompt)

        results = self.llm.generate(prompts)
        scores = self.parse_llm_result(prompts, results)
        return scores
