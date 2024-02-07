from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import Dataset
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from rageval.metrics.base import MetricWithLLM
from rageval.models.openai import OpenAILLM
from rageval.utils.utility import json_loader
from rageval.utils import CONTEXT_RECALL_RA


@dataclass
class ContextRecall(MetricWithLLM):
    """
    Estimates context recall by estimating TP and FN using annotated answer and retrieved context.

    Attributes
    ----------
    name : str
    batch_size : int, Batch size for openai completion.

    """

    name: str = "context_recall"  # type: ignore
    batch_size: int = 15

    def init_model(self):
        """Initializee the LLM model with OpenAILLM."""
        self.llm: OpenAILLM = OpenAILLM('gpt-3.5-turbo-16k', 'OPENAI_API_KEY')

    def _score_batch(
        self,
        dataset: Dataset,
    ) -> list:
        prompts = []
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

        responses: list[list[str]] = []
        results = self.llm.generate(prompts)
        responses = [[i.text for i in r] for r in results.generations]
        results = []
        # for each question-answer pair
        for response in responses:
            response = json_loader.safe_load(response[0], self.llm)
            # response: list of dict; each dict is a statement extracted from gt_answer
            if response:
                reasonings = [
                    str(item)
                    for item in response
                ]
                scores = [
                    int(item.get("Attributed", "0").strip() == "1")
                    if item.get("Attributed")
                    else np.nan
                    for item in response
                ]
                data = {'reasoning': reasonings, 'score': scores}
                results.append(pd.DataFrame(data))
            else:
                data = {'reasoning': [], 'score': [np.nan]}
                results.append(pd.DataFrame(data))

        df = pd.concat(results)
        print(df['score'].mean())
        return df['score'].mean(), df
