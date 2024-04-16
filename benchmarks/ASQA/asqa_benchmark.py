from typing import Dict, Tuple, Any, Optional
from datasets import Dataset
import math
import os
import argparse
from benchmarks import BaseBenchmark
from rageval.metrics import (AnswerRougeCorrectness, AnswerEMCorrectness, AnswerDisambigF1Correctness)


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

    def is_existed(self, column_name: str) -> bool:
        """Check if the column exists in the dataset."""
        return column_name in self.dataset.column_names

    def _evaluate(self, ) -> Tuple[Dict[Any, Any], Dataset]:
        """Evaluate the dataset and return the dataset with scores.

        For the ASQA dataset, the `short_answers` and `long_answers` are stored in the "qa_pairs" and "annotations" columns, respectively. We need to extract them and add them to the dataset.

        We use the `short_answers` as the `gt_answers` to evaluate the string Exact Match correctness and the `long_answers` to evaluate the RougeL and DisambigF1 score. And then we calculate the `DR score` as the geometric mean of the RougeL and DisambigF1 scores.
        """
        if not self.is_existed("short_answers"):
            self.dataset = self.dataset.map(lambda example: {"short_answers": [ann["short_answers"] for ann in example["qa_pairs"]]})
        if not self.is_existed("long_answers"):
            self.dataset = self.dataset.map(lambda example: {"long_answers": [ann["long_answer"] for ann in example["annotations"]]})

        ground_truths = {
            "answer_disambig_f1": ("long_answers", "gt_answers"),
            "answer_rouge_correctness": ("long_answers", "gt_answers"),
            "answer_exact_match": ("short_answers", "gt_answers")
        }

        results = {}
        for m in self.metrics:
            if m.name in ground_truths:
                print(f"Calculating {m.name}...")

                if self.is_existed(m.name):
                    # Remove the metric column if it already exists
                    self.dataset = self.dataset.remove_columns(m.name)
                if not self.is_existed(ground_truths[m.name][0]):
                    # Check if the ground truth column exists
                    raise ValueError(f"The column {ground_truths[m.name][0]} is not in the dataset. Please check the column names.")

                # Rename the ground truth column for metric calculation
                self.dataset = self.dataset.rename_column(*ground_truths[m.name])
                # Compute the metric
                results[m.name], self.dataset = m.compute(self.dataset, self.batch_size)
                # Rename the column back
                self.dataset = self.dataset.rename_column(*ground_truths[m.name][::-1])
                # Add the ground truth column name
                self.dataset = self.dataset.map(lambda example: {f"{m.name}.{ground_truths[m.name][1]}": ground_truths[m.name][0]})

        if self.is_existed("answer_rouge_correctness") and self.is_existed("answer_disambig_f1"):
            if self.is_existed("DR_score"):
                self.dataset = self.dataset.remove_columns("DR_score")
            print("Calculating DR score...")
            def dr_score(d:dict):
                d['DR_score'] = math.sqrt(d["answer_disambig_f1"] * d["answer_rouge_correctness"])
                return d
            self.dataset = self.dataset.map(dr_score)
            results["DR_score"] = math.sqrt(results["answer_disambig_f1"] * results["answer_rouge_correctness"])

        return results, self.dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".rageval/benchmark")
    parser.add_argument("--split", type=str, default="mistral_7b")
    args = parser.parse_args()

    benchmark = ASQABenchmark()

    results = benchmark.evaluate(path="golaxy/rag-bench", name="asqa", split=args.split)
    print(f"Results:\n {results}")

    benchmark.save_results(os.path.join(args.output_dir,"results", f"{args.split}.jsonl"))
    benchmark.save_dataset(os.path.join(args.output_dir,"dataset", f"{args.split}.jsonl"))

    benchmark.set_metric([AnswerEMCorrectness(ignore_case=False)])
    results = benchmark.evaluate()
    print(f"Results:\n {results}")
