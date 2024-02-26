# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
from abc import ABC
from typing import List, Any, Callable


import datasets
from datasets import Dataset
from dataclasses import dataclass
from langchain.schema import LLMResult

from rageval.metrics import MetricWithLLM, add_attribute
from rageval.utils.prompt import REJECT_RATE_PROMPT


_DESCRIPTION = """\
ContextRejectRate is the metric to meature the unknown robustness of LLM based 
on the given context. 

"""

_KWARGS_DESCRIPTION = """\
"""


@dataclass
@add_attribute('mtype', 'ContextRejectRate')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ContextRejectRate(MetricWithLLM):

    name = "context_reject_rate"

    def __init__(self):
        """Explicitly initialize the ContextRejectRate to ensure all parent class initialized."""
        self._required_columns = ['questions', 'contexts']
        super().__init__()

    def init_model(self, model: Callable):
        """Initializee the LLM model."""
        self.model = model

    def parse_llm_result(self, prompts: str, result: LLMResult):
        """Parse the LLM Result based on the Prompt."""
        num_reject = 0
        num = 0
        responses = [[i.text for i in r] for r in result.generations]
        # for each question-answer pair
        for response in responses:
            answer = response.split("Answer:")[1]
            if "sorry, cannot answer the question" in answer:
                num_reject = num_reject + 1
                num = num + 1
            else:
                num = num + 1
        score = num_reject/num
        return score

    def _compute_batch(
        self,
        dataset: Dataset,
    ) -> float:
        prompts = []
        questions, contexts = (
            dataset["questions"],
            dataset["contexts"],
        )

        prompts = []
        for question_, context in zip(questions, contexts):
            prompt = REJECT_RATE_PROMPT.format(
                question=question_, evidence=context
            )
            prompts.append(prompt)
        
        result = self.llm.generate(prompts)
        score = self.parse_llm_result(prompts, result)
        return score