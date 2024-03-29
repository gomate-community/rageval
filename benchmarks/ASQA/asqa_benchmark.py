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

    def prepare_data(self, metric: str, label_column: str, input_column: str):
        """Modify self.dataset for different metric. Remove the existing metric column for metric to be evaluated and add the label column will be uesd.

        Args:
            input_column: The column name of the input text that has already existed in self.dataset, e.g. `long_answer`.
            label_column: The column name of the label text that the metric requires, e.g. `gt_answer`.
        """
        if metric in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns(metric)

        if input_column not in self.dataset.column_names:
            raise ValueError(f"The input column {input_column} is not in the dataset. Please check the column names.")

        self.dataset = self.dataset.map(lambda example: {label_column: example[input_column]}, batched=True)

    def _evaluate(self, ) -> Tuple[Dict[Any, Any], Dataset]:
        """Evaluate the dataset and return the dataset with scores.

        For the ASQA dataset, the `short_answers` and `long_answers` are stored in the "qa_pairs" and "annotations" columns, respectively. We need to extract them and add them to the dataset.

        We use the `short_answers` as the `gt_answers` to evaluate the string Exact Match correctness and the `long_answers` to evaluate the RougeL and DisambigF1 score. And then we calculate the `DR score` as the geometric mean of the RougeL and DisambigF1 scores.
        """
        if "short_answers" not in self.dataset.column_names:
            self.dataset = self.dataset.map(lambda example: {"short_answers": [ann["short_answers"] for ann in example["qa_pairs"]]})
        if "long_answers" not in self.dataset.column_names:
            self.dataset = self.dataset.map(lambda example: {"long_answers": [ann["long_answer"] for ann in example["annotations"]]})

        ground_truths = {
            "answer_disambig_f1": ("gt_answers", "long_answers"),
            "answer_rouge_correctness": ("gt_answers", "long_answers"),
            "answer_exact_match": ("gt_answers", "short_answers")
        }

        results = {}
        for m in self.metrics:
            if m.name in ground_truths:
                print(f"Calculating {m.name}...")
                self.prepare_data(m.name, *ground_truths[m.name])
                results[m.name], self.dataset = m.compute(self.dataset, self.batch_size)
                self.dataset = self.dataset.map(lambda example: {f"{m.name}.{ground_truths[m.name][0]}": ground_truths[m.name][1]}) # Add the ground truth column name

        if "gt_answers" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("gt_answers")

        if "answer_rouge_correctness" in self.dataset.column_names and "answer_disambig_f1" in self.dataset.column_names and "DR_score" not in self.dataset.column_names:
            print("Calculating DR score...")
            def dr_score(d:dict):
                d['DR_score'] = math.sqrt(d["answer_disambig_f1"] * d   ["answer_rouge_correctness"])
                return d
            self.dataset = self.dataset.map(dr_score)
            results["DR_score"] = math.sqrt(results["answer_disambig_f1"] * results ["answer_rouge_correctness"])

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
