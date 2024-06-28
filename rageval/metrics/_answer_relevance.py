from __future__ import annotations

import logging
import typing as t

import numpy as np
from dataclasses import dataclass, field

from ragas.llms.output_parser import RagasoutputParser, _output_parser
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from ragas.llms.prompt import PromptValue


class AnswerRelevanceClassification:
    question: str
    noncommittal: int


class AnswerRelevancy(MetricWithLLM, MetricWithEmbeddings):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    question_generation: Prompt
    strictness: int = 3

    def __init__(self):
        self.question_generation = Prompt(
            name="question_generation",
            instruction="""Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers""",
            output_format_instruction=None,
            examples=[],
            input_keys=["answer", "context"],
            output_key="output",
            output_type="json",
            language="english",
        )

        self.sentence_segmenter = None
        self._reproducibility = 1

    @property
    def reproducibility(self):
        return self._reproducibility

    @reproducibility.setter
    def reproducibility(self, value):
        if value < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            value = 1
        elif value % 2 == 0:
            logger.warning(
                "reproducibility level cannot be set to even number, setting to odd"
            )
            value += 1
        self._reproducibility = value

    def calculate_similarity(
        self, question: str, generated_questions: list[str]
    ):
        assert self.embeddings is not None
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            ) / norm
        )

    def _calculate_score(
        self, answers: t.Sequence[AnswerRelevanceClassification], row: t.Dict
    ) -> float:
        question = row["question"]
        gen_questions = [answer.question for answer in answers]
        committal = np.any([answer.noncommittal for answer in answers])
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            cosine_sim = self.calculate_similarity(question, gen_questions)
            score = cosine_sim.mean() * int(not committal)

        return score

    def _create_question_gen_prompt(self, row: t.Dict) -> PromptValue:
        ans, ctx = row["answer"], row["contexts"]
        return self.question_generation.format(answer=ans, context="\n".join(ctx))

    async def _ascore(self, row: t.Dict, callbacks: Callbacks, is_async: bool) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt = self._create_question_gen_prompt(row)
        result = await self.llm.generate(
            prompt,
            n=self.strictness,
            callbacks=callbacks,
            is_async=is_async,
        )

        answers = [
            await _output_parser.aparse(result.text, prompt, self.llm)
            for result in result.generations[0]
        ]
        if any(answer is None for answer in answers):
            return np.nan

        answers = [answer for answer in answers if answer is not None]
        return self._calculate_score(answers, row)

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logger.info(f"Adapting AnswerRelevancy metric to {language}")
        self.question_generation = self.question_generation.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.question_generation.save(cache_dir)


answer_relevancy = AnswerRelevancy()
