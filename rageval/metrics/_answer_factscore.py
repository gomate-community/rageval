import os
from dataclasses import dataclass
from typing import List
import datasets

from rageval.metrics import Metric, add_attribute, FactScorer

_DESCRIPTION = """\
FactScore evaluates the factual correctness between predicted and ground truth texts using a knowledge-based scoring approach.
"""

_KWARGS_DESCRIPTION = """\
Args:
    name : str
    items : list, List of tuples containing (gold, text, pred) where gold is the ground truth, text is the input text, and pred is the predicted text.

Returns:
    float: FactScore representing the average factual correctness score.

Examples:
    Example 1:
    >>> items = [('ground_truth', 'input_text', 'predicted_text')]
    >>> metric = FactScore()
    >>> score, results = metric.compute(items)
"""

_CITATION = """
@misc{xie2023pixiu,
    title={PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance},
    author={Qianqian Xie and Weiguang Han and Xiao Zhang and Yanzhao Lai and Min Peng and Alejandro Lopez-Lira and Jimin Huang},
    year={2023},
    eprint={2306.05443},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


@dataclass
@add_attribute('mtype', 'FactualCorrectness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class FactScore(Metric):
    """FactScore metric for evaluating factual correctness."""

    name = "fact_score"

    ALIAS = ['fact_score']

    def __init__(self):
        """
        Explicitly initialize FactScore.

        Ensure all parent classes are initialized.
        """
        super().__init__()

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "items": datasets.Value("list", description="List of tuples (gold, text, pred)"),
                }
            ),
            codebase_urls=[],
            reference_urls=[]
        )

    def compute(self, items: List[tuple]) -> tuple:
        """Compute FactScore."""
        golds, texts, preds = zip(*items)

        fs = FactScorer("retrieval+ChatGPT", openai_key=os.environ["OPENAI_API_KEY"])
        fs.register_knowledge_source("finterms", data_path="./src/factscore_package/.cache/finterms.jsonl",
                                     db_path="./src/factscore_package/.cache/fin_terms.db")

        score = 0
        num_facts = 0
        for i in range(len(texts)):
            try:
                out = fs.get_score([texts[i]], [preds[i]], knowledge_source="finterms")
                score += out["score"] * out["num_facts_per_response"]
                num_facts += out["num_facts_per_response"]
            except Exception as e:
                print(f"Error occurred: {e}")

        if num_facts > 0:
            average_score = score / num_facts
        else:
            average_score = 0.0

        return average_score, None  # No detailed results to return in this case
