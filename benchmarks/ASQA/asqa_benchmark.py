from typing import Dict, Tuple, Any, Optional
from datasets import Dataset, load_dataset
import json
import math
import os
import logging
import argparse
from benchmarks import BaseBenchmark
from rageval.metrics import (AnswerRougeCorrectness, AnswerEMCorrectness, AnswerDisambigF1Correctness)


logger = logging.getLogger(__name__)

class ASQABenchmark(BaseBenchmark):
    """Benchmark for ASQA dataset.

    The ASQA dataset is a question-answering dataset that contains factoid questions and long-form answers. The benchmark evaluates the correctness of the answers in the dataset.
    """

    name = "asqa_benchmark"
    metrics = [AnswerRougeCorrectness(rouge_type="rougeL"), 
               AnswerEMCorrectness(ignore_case=True), 
               AnswerDisambigF1Correctness()]

    def __init__(self) -> None:
        """Initialization."""
        super().__init__()

    def load_data(self, **kwargs):
        """Load ASQA dataset.

        For the ASQA dataset, the `short_answers` and `long_answers` are stored in the "qa_pairs" and "annotations" columns, respectively. We need to extract them and add them to the dataset.
        """
        print("Load ASQA dataset...")
        super().load_data(**kwargs)
        if "short_answers" not in self.dataset.features:
            self.dataset = self.dataset.map(lambda example: {"short_answers": [ann["short_answers"] for ann in example["qa_pairs"]]})
        if "long_answers" not in self.dataset.features:
            self.dataset = self.dataset.map(lambda example: {"long_answers": [ann["long_answer"] for ann in example["annotations"]]})
        print("ASQA dataset loaded.")

    def prepare_data(self, label_column: str, input_column: str):
        """Modify self.dataset for different metric.

        Args:
            input_column: The column name of the input text that has already existed in self.dataset, e.g. `long_answer`.
            label_column: The column name of the label text that the metric requires, e.g. `gt_answer`.
        """
        if input_column not in self.dataset.column_names:
            raise ValueError(f"The input column {input_column} is not in the dataset. Please check the column names.")

        if label_column in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns(label_column)
        self.dataset = self.dataset.add_column(label_column, self.dataset[input_column])

    def _evaluate(self, ) -> Tuple[Dict[Any, Any], Dataset]:
        """Evaluate the dataset and return the dataset with scores.

        We use the `short_answers` as the `gt_answers` to evaluate the string Exact Match correctness and the `long_answers` to evaluate the RougeL and DisambigF1 score. And then we calculate the `DR score` as the geometric mean of the RougeL and DisambigF1 scores.
        """
        print("Start evaluate...")

        ground_truths = {
            "answer_disambig_f1": ("gt_answers", "long_answers"),
            "answer_rouge_correctness": ("gt_answers", "long_answers"),
            "answer_exact_match": ("gt_answers", "short_answers")
        }
        results = {}
        scores = {}
        for m in self.metrics:
            if m.name in ground_truths:
                print(f"Evaluating {m.name}...")
                self.prepare_data(*ground_truths[m.name])
                results[m.name], self.dataset = m.compute(self.dataset, self.batch_size)
                self.dataset = self.dataset.map(lambda example: {f"{m.name}.{ground_truths[m.name][0]}": ground_truths[m.name][1]}) # Add the ground truth column name
                scores[m.name] = self.dataset[m.name]

        if "gt_answers" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("gt_answers")
        # scores = [{k:v[i] for k,v in scores.items()} for i in range(len(self.dataset))]
        # self.dataset = self.dataset.add_column("scores", scores)

        if "answer_rouge_correctness" in self.dataset.column_names and "answer_disambig_f1" in self.dataset.column_names and "DR_score" not in self.dataset.column_names:
            print("Calculating DR score...")
            def dr_score(d:dict):
                d['DR_score'] = math.sqrt(d["answer_disambig_f1"] * d   ["answer_rouge_correctness"])
                return d
            self.dataset = self.dataset.map(dr_score)
            results["DR_score"] = math.sqrt(results["answer_disambig_f1"] * results ["answer_rouge_correctness"])

        print("Evaluation finished.")

        return results, self.dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="benchmarks/ASQA/output")
    args = parser.parse_args()

    benchmark = ASQABenchmark()

    results = benchmark.evaluate(path="json", data_files=os.path.join(args.output_dir, "dataset.jsonl"), split="train")
    print(f"Results:\n {results}")

    benchmark.save_results(os.path.join(args.output_dir, "results.jsonl"))
    benchmark.save_dataset(os.path.join(args.output_dir, "result_dataset.jsonl"))

    benchmark.dataset = benchmark.dataset.remove_columns("answer_exact_match")
    benchmark.set_metric([AnswerEMCorrectness(ignore_case=False)])
    results = benchmark.evaluate()
    print(f"Results:\n {results}")
