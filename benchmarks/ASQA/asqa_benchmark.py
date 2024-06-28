from typing import Dict, Tuple, Any, Optional
from datasets import Dataset
import numpy as np
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

    ground_truths = {
        "answer_disambig_f1": "long_answers",
        "answer_rouge_correctness": "long_answers",
        "answer_exact_match": "short_answers"
    }

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

        results = {}
        for m in self.metrics:
            if m.name in self.ground_truths:
                print(f"Calculating {m.name}...")

                if self.is_existed(m.name):
                    # Remove the metric column if it already exists
                    self.dataset = self.dataset.remove_columns(m.name)
                if not self.is_existed(self.ground_truths[m.name]):
                    # Check if the ground truth column exists
                    raise ValueError(f"The column {self.ground_truths[m.name]} is not in the dataset. Please check the column names.")

                avg_scores, scores = m.compute(
                    self.dataset["answers"], 
                    self.dataset[self.ground_truths[m.name]]
                )
                results[m.name] = avg_scores
                self.dataset = self.dataset.add_column(m.name, scores)

                print(f"{m.name}: {avg_scores}")

        if self.is_existed("answer_rouge_correctness") and self.is_existed("answer_disambig_f1"):
            # Notice that DR score is an overall geometric mean of RougeL and DisambigF1 scores, which is calculated as sqrt(RougeL * DisambigF1) for whole dataset instead of average of each sample.
            print("Calculating DR score...")
            results["DR_score"] = np.sqrt(np.average(self.dataset["answer_disambig_f1"]) * np.average(self.dataset["answer_rouge_correctness"]))
            print(f"DR_score: {results['DR_score']}")

        return results, self.dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".rageval/benchmark")
    parser.add_argument("--split", type=str, default="llama2_7b_chat")
    args = parser.parse_args()

    benchmark = ASQABenchmark()

    results = benchmark.evaluate(path="golaxy/rag-bench", name="asqa", split=args.split)
    print(f"Results:\n {results}")

    benchmark.save_results(os.path.join(args.output_dir,"results", f"{args.split}.jsonl"))
    benchmark.save_dataset(os.path.join(args.output_dir,"dataset", f"{args.split}.jsonl"))

    benchmark.set_metric([AnswerEMCorrectness(ignore_case=False)])
    results = benchmark.evaluate()
    print(f"Results:\n {results}")
