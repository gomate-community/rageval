from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from rageval.metrics.base import MetricWithLLM
from rageval.models.openai import OpenAILLM
from rageval.utils.utility import json_loader


CONTEXT_RECALL_RA = HumanMessagePromptTemplate.from_template(
    """
Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only "Yes" (1) or "No" (0) as a binary classification. Output json with reason.


question: What can you tell me about albert Albert Einstein?
context: Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
answer: Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895
classification:
[
    {{  "statement_1":"Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
        "reason": "The date of birth of Einstein is mentioned clearly in the context.",
        "Attributed": "1"
    }},
    {{
        "statement_2":"He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics.",
        "reason": "The exact sentence is present in the given context.",
        "Attributed": "1"
    }},
    {{
        "statement_3": "He published 4 papers in 1905.",
        "reason": "There is no mention about papers he wrote in the given context.",
        "Attributed": "0"
    }},
    {{
        "statement_4":"Einstein moved to Switzerland in 1895.",
        "reason": "There is no supporting evidence for this in the given context.",
        "Attributed": "0"
    }}
]

question: who won 2020 icc world cup?
context: Who won the 2022 ICC Men's T20 World Cup?
The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.
answer: England
classification:
[
    {{
        "statement_1":"England won the 2022 ICC Men's T20 World Cup.",
        "reason": "From context it is clear that England defeated Pakistan to win the World Cup.",
         "Attributed": "1"
    }}
]

question:{question}
context:{context}
answer:{answer}
classification:
"""  # noqa: E501
)


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
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []
        question, ground_truths, contexts = (
            dataset["question"],
            dataset["ground_truths"],
            dataset["contexts"],
        )

        for qstn, gt, ctx in zip(question, ground_truths, contexts):
            gt = "\n".join(gt) if isinstance(gt, list) else gt
            ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx
            human_prompt = CONTEXT_RECALL_RA.format(
                question=qstn, context=ctx, answer=gt
            )
            prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

        responses: list[list[str]] = []
        results = self.llm.generate(prompts)
        responses = [[i.text for i in r] for r in results.generations]
        scores = []
        for response in responses:
            response = json_loader.safe_load(response[0], self.llm)
            if response:
                response = [
                    int(item.get("Attributed", "0").strip() == "1")
                    if item.get("Attributed")
                    else np.nan
                    for item in response
                ]
                denom = len(response)
                numerator = sum(response)
                scores.append(numerator / denom)
            else:
                scores.append(np.nan)

        return scores


context_recall = ContextRecall()
