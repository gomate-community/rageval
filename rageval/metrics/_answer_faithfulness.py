from __future__ import annotations

import logging
import typing as t
import numpy as np
from dataclasses import dataclass, field

from ragas.llms.output_parser import _statements_output_parser, _faithfulness_output_parser, RagasoutputParser
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM, ensembler, get_segmenter

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from ragas.llms.prompt import PromptValue

logger = logging.getLogger(__name__)


class StatementFaithfulnessAnswer:
    statement: str
    reason: str
    verdict: int


class StatementFaithfulnessAnswers:
    __root__: t.List[StatementFaithfulnessAnswer]


class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    nli_statements_message: Prompt
    statement_prompt: Prompt
    max_retries: int = 1

    def __init__(self):
        self.nli_statements_message = Prompt(
            name="nli_statements",
            instruction="Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.",
            output_format_instruction=None,
            examples=[],
            input_keys=["context", "statements"],
            output_key="answer",
            output_type="json",
            language="english",
        )

        self.statement_prompt = Prompt(
            name="long_form_answer",
            output_format_instruction=None,
            instruction="Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON.",
            examples=[],
            input_keys=["question", "answer", "sentences"],
            output_key="analysis",
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

    def __post_init__(self):
        if self.sentence_segmenter is None:
            language = self.nli_statements_message.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def _compute_score(self, answers: StatementFaithfulnessAnswers):
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.__root__
        )
        num_statements = len(answers.__root__)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _ascore(
        self: t.Self, row: t.Dict, callbacks: Callbacks, is_async: bool
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        p_value = self._create_statements_prompt(row)
        statements = await self.llm.generate(
            p_value,
            callbacks=callbacks,
            is_async=is_async,
        )
        statements = await _statements_output_parser.aparse(
            statements.generations[0][0].text, p_value, self.llm, self.max_retries
        )

        if statements is None:
            return np.nan

        statements = [item["simpler_statements"] for item in statements.dicts()]
        statements = [item for sublist in statements for item in sublist]

        assert isinstance(statements, t.List), "statements must be a list"

        p_value = self._create_nli_prompt(row, statements)
        nli_result = await self.llm.generate(
            p_value,
            callbacks=callbacks,
            is_async=is_async,
            n=self._reproducibility,
        )

        nli_result_text = [
            nli_result.generations[0][i].text for i in range(self._reproducibility)
        ]
        faithfulness_list = [
            await _faithfulness_output_parser.aparse(
                text, p_value, self.llm, self.max_retries
            )
            for text in nli_result_text
        ]

        faithfulness_list = [
            faith.dicts() for faith in faithfulness_list if faith is not None
        ]

        if faithfulness_list:
            faithfulness_list = ensembler.from_discrete(
                faithfulness_list,
                "verdict",
            )

            faithfulness_list = StatementFaithfulnessAnswers.parse_obj(
                faithfulness_list
            )
        else:
            return np.nan

        return self._compute_score(faithfulness_list)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logger.info(f"Adapting Faithfulness metric to {language}")

        self.nli_statements_message = self.nli_statements_message.adapt(
            language, self.llm, cache_dir
        )
        self.statement_prompt = self.statement_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.nli_statements_message.save(cache_dir)


faithfulness = Faithfulness()
