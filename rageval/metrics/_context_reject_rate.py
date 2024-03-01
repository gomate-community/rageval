from dataclasses import dataclass
from typing import Callable

import datasets
from datasets import Dataset
from langchain.schema import LLMResult

from rageval.metrics import Metric, add_attribute
from rageval.utils.prompt import REJECT_RATE_PROMPT

_DESCRIPTION = """\
ContextRejectRate is the metric to measure the unknown robustness of LLM based on the given context.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    batch_size : int, Batch size for openai completion.

Optional Args:
    None

Functions:
    parse_llm_result: parse the results of LLM
    _compute_batch: compute the score by measure how many rejected answers in all answers.
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
@add_attribute('mtype', 'ContextRejectRate')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ContextRejectRate(Metric):

    name = "context_reject_rate"

    ALIAS = ['context_reject_rate']

    def __init__(self, model: Callable):
        """Explicitly initialize the ContextRejectRate to ensure all parent class initialized."""
        self._required_columns = ['questions', 'contexts']
        self.model = model
        super().__init__()

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
                    "questions": datasets.Value("string"),
                    "contexts": datasets.Sequence(datasets.Value("string"))
                }
            ),
            codebase_urls=[],
            reference_urls=[]
        )

    def parse_llm_result(self, result: LLMResult):
        """Parse the LLM Result based on the Prompt."""
        responses = [[i.text for i in r] for r in result.generations]
        scores = []
        # for each question-answer pair
        for response in responses:
            response = response[0]
            answer = response.split("Answer:")[1]
            if "sorry, cannot answer the question" in answer:
                scores.append(1.)
            else:
                scores.append(0.)
        return scores

    def _compute_batch(
        self,
        dataset: Dataset,
    ) -> list:
        """Compute the score by measure how many rejected answers in all answers."""
        questions, contexts = (
            dataset["questions"],
            dataset["contexts"],
        )

        prompts = []
        for question, context in zip(questions, contexts):
            prompt = REJECT_RATE_PROMPT.format(
                question=question, evidence=context
            )
            prompts.append(prompt)

        results = self.model.generate(prompts)
        scores = self.parse_llm_result(results)
        return scores
