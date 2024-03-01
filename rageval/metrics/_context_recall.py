import typing
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import Dataset
from langchain.schema import LLMResult

from rageval.metrics import MetricWithLLM, add_attribute
from rageval.utils.utility import json_loader
from rageval.utils import CONTEXT_RECALL_RA


@dataclass
@add_attribute('mtype', 'ContextRelevancy')
class ContextRecall(MetricWithLLM):
    """
    Estimates context recall by estimating TP and FN using annotated answer and retrieved context.

    Attributes
    ----------
    name : str

    """

    name = "context_recall"

    def __init__(self, model: typing.Callable):
        """Initialize the LLM model."""
        self.llm = model
        self._required_columns = ['questions', 'gt_answers', 'contexts']

    def parse_llm_result(self, prompts: str, result: LLMResult):
        """
        Parse the LLM Result based on the Prompt.

        TODO: use prompts to parse the result.
        """
        results = []
        scores = []
        responses = [[i.text for i in r] for r in result.generations]
        # for each question-answer pair
        for response in responses:
            response = json_loader.safe_load(response[0], self.llm)
            # response: list of dict; each dict is a statement extracted from gt_answer
            if response:
                reasonings = [
                    str(item)
                    for item in response
                ]
                score = [
                    int(item.get("Attributed", "0").strip() == "1")
                    if item.get("Attributed")
                    else np.nan
                    for item in response
                ]
                data = {'reasoning': reasonings, 'score': score}
                scores.append(np.average(score))
            else:
                data = {'reasoning': [np.nan], 'score': [0.]}
                scores.append(0.)
            results.append(pd.DataFrame(data))
        # Note that the `results can be recorded by logger.info`
        return scores

    def _compute_batch(
        self,
        dataset: Dataset,
    ) -> list:
        question, ground_truths, contexts = (
            dataset["questions"],
            dataset["gt_answers"],
            dataset["contexts"],
        )

        prompts = []
        for qstn, gt, ctx in zip(question, ground_truths, contexts):
            gt = "\n".join(gt) if isinstance(gt, list) else gt
            ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx
            prompt = CONTEXT_RECALL_RA.format(
                question=qstn, context=ctx, answer=gt
            )
            prompts.append(prompt)

        result = self.llm.generate(prompts)
        scores = self.parse_llm_result(prompts, result)
        return scores
