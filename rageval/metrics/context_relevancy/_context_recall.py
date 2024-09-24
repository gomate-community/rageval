from dataclasses import dataclass
from typing import Callable, List, Tuple

import datasets
import numpy as np
import pandas as pd
from langchain.schema import LLMResult
from tqdm import tqdm

from rageval.metrics import MetricWithLLM, add_attribute
from rageval.utils.utility import json_loader
from rageval.utils import CONTEXT_RECALL_RA

_DESCRIPTION = """\
ContextRecall evaluates contexts relevancy based on gt_answers.

For details, see the doc: https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html.
"""

_KWARGS_DESCRIPTION = r"""\
Args:
    name : str
    batch_size : int, Batch size for openai completion.
    model : Callable, The LLM model to use.

Optional Args:
    None

Functions:
    parse_llm_result: Parse the LLM Result based on the Prompt
    _compute_batch: Compute the score by measure whether the args:`contexts` can be supported by args:`gt_answers`.

Examples:
    >>> from datasets import Dataset
    >>> from langchain.llms.fake import FakeListLLM
    >>> import rageval as rl
    >>> sample = {
    ...     "questions": ["恐龙是怎么被命名的？"],
    ...     "gt_answers": [["1841年，英国科学家理查德·欧文在研究几块样子像蜥蜴骨头化石时，认为它们是某种史前动物留下来的，并命名为恐龙，意思是“恐怖的蜥蜴”。"]],
    ...     "contexts": [["[12]恐龙是 介于冷血和温血之间的动物2014年6月，有关恐龙究竟是像鸟类和哺乳动物一样的温血动物，还是类似爬行动物、鱼类和两栖动物的冷血动物的问题终于有了答案——恐龙其实是介于冷血"
    ...                   "和温血之间的动物。 [12]“我们的结果显示恐龙所具有的生长速率和新陈代谢速率，既不是冷血生物体也不是温血生物体所具有的特征。它们既不像哺乳动物或者鸟类，也不像爬行动物或者鱼类，"
    ...                   "而是介于现代冷血动物和温血动物之间。简言之，它们的生理机能在现代社会并不常见。”美国亚利桑那大学进化生物学家和生态学家布莱恩·恩奎斯特说。墨西哥生物学家表示，正是这种中等程度的"
    ...                   "新陈代谢使得恐龙可以长得比任何哺乳动物都要大。温血动物需要大量进食，因此它们频繁猎捕和咀嚼植物。“很难想象霸王龙大小的狮子能够吃饱以 存活下来。","[12]哺乳动物起源于爬行动物，"
    ...                   "它们的前身是“似哺乳类的爬行动物”，即兽孔目，早期则是“似爬行类的哺乳动物”，即哺乳型动物。 [12]中生代的爬行动物，大部分在中生代的末期灭绝了；一部分适应了变化的环境被保留下来，"
    ...                   "即现存的爬行动物（如龟鳖类、蛇类、鳄类等）；还有一部分沿着不同的进化方向，进化成了现今的鸟类和哺乳类。 [12]恐龙是 介于冷血和温血之间的动物2014年6月，有关恐龙究竟是像鸟类和"
    ...                   "哺乳动物一样的温血动物，还是类似爬行动物、鱼类和两栖动物的冷血动物的问题终于有了答案——恐龙其实是介于冷血和温血之间的动物。"
    ...                 ]]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> model = FakeListLLM(
    ...     responses=['[\n    {\n        "statement_1":"恐龙的命名始于1841年，由英国科学家理查德·欧文命名。",\n        "reason": "The answer '
    ...                'provides the exact year and the scientist who named the dinosaurs.",\n        "Attributed": "1"'
    ...                '\n    },\n    {\n        "statement_2":"欧文在研究几块样子像蜥蜴骨头化石时，认为它们是某种史前动物留下来的，并命名为恐龙。",'
    ...                '\n        "reason": "The answer accurately describes the process of how dinosaurs were named.",'
    ...                '\n        "Attributed": "1"\n    }\n]'
    ...               ]
    ... )
    >>> metric = rl.metrics.ContextRecall(model)
    >>> metric.mtype
    'ContextRelevancy'
    >>> score, results = metric.compute(dataset['questions'], dataset['gt_answers'], dataset['contexts'], 1)
    >>> assert 0 <= score <= 1
"""

_CITATION = """
@misc{ragas,
    author= {explodinggradients},
    year  = {2023},
    title = {ragas},
    note  = {https://github.com/explodinggradients/ragas, Last accessed on 2024-3-2},
}
"""


@dataclass
@add_attribute('mtype', 'ContextRelevancy')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ContextRecall(MetricWithLLM):
    """Estimates context recall by estimating TP and FN using annotated answer and retrieved context."""

    name = "context_recall"

    ALIAS = ['context_recall']

    def __init__(self, model: Callable):
        """Explicitly initialize the AnswerEMCorrectness to ensure all parent class initialized."""
        super().__init__(model)

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
                    "gt_answers": datasets.Sequence(datasets.Value("string")),
                    "contexts": datasets.Sequence(datasets.Value("string"))
                }
            ),
            codebase_urls=["https://github.com/explodinggradients/ragas"],
            reference_urls=["https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html"]
        )

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

    def compute(
        self,
        questions: List[str],
        ref_answers: List[str],
        contexts: List[List[str]],
        batch_size: int,
    ) -> Tuple[float, List[float]]:
        """Evaluate the dataset."""
        scores = []
        length = len(questions)
        if batch_size:
            for start in tqdm(range(0, length, batch_size)):
                end = start + batch_size
                end = end if end < length else length
                score = self._compute_batch(
                    questions[start:end],
                    ref_answers[start:end],
                    contexts[start:end]
                )
                scores.extend(score)
        else:
            scores = self._compute_batch(questions, ref_answers, contexts)

        return np.average(scores), scores

    def _compute_one():
        pass

    def _compute_batch(
        self,
        questions: List[str],
        ref_answers: List[str],
        contexts: List[List[str]]
    ) -> List[float]:

        prompts = []
        for question, ref_answer, context in zip(questions, ref_answers, contexts):
            ref_answer = "\n".join(ref_answer) if isinstance(ref_answer, list) else ref_answer
            context = "\n".join(context) if isinstance(context, list) else context
            prompt = CONTEXT_RECALL_RA.format(
                question=question,
                context=context,
                answer=ref_answer
            )
            prompts.append(prompt)

        result = self.llm.generate(prompts)
        scores = self.parse_llm_result(prompts, result)

        return scores
