from datasets import Dataset, load_dataset
import json
import math
import os
import logging
import argparse
from benchmarks import BaseBenchmark


logger = logging.getLogger(__name__)

class ASQABenchmark(BaseBenchmark):

    name = "asqa_benchmark"
    metrics = ["AnswerRougeCorrectness", "AnswerEMCorrectness", "AnswerDisambigF1Correctness"]

    def __init__(self, output_dir: str, batch_size: int = 1) -> None:
        self.output_dir = output_dir
        self.batch_size = batch_size

    def load_data(self, **kwargs) -> Dataset:
        """Load ASQA dataset.

        For the ASQA dataset, the short_answers and long_answers are stored in the "qa_pairs" and "annotations" columns, respectively.
        """
        print("Load ASQA dataset...")
        dataset = load_dataset(**kwargs)
        dataset = dataset.map(lambda example: {"short_answers": [ann["short_answers"] for ann in example["qa_pairs"]]})
        dataset = dataset.map(lambda example: {"long_answers": [ann["long_answer"] for ann in example["annotations"]]})
        print("ASQA dataset loaded.")
        return dataset

    def evaluate(self, dataset: Dataset) -> Dataset:
        """Evaluate the dataset and return the dataset with scores.

        We use the `short_answers` to evaluate the string Exact Match correctness and the `long_answers` to evaluate the RougeL and DisambigF1 score. And then we calculate the `DR score` as the geometric mean of the two scores.
        """
        print("Start evaluate...")
        self.results = {}
        scores = {}
        for m in self.metrics:
            if m == "AnswerRougeCorrectness":
                print("Evaluating AnswerRougeCorrectness...")
                metric = self.get_metric(m, rouge_type="rougeL")
                ds = dataset.add_column("gt_answers", dataset["long_answers"])
            elif m == "AnswerEMCorrectness":
                print("Evaluating AnswerEMCorrectness...")
                metric = self.get_metric(m, ignore_case=True)
                ds = dataset.add_column("gt_answers", dataset["short_answers"])
            elif m == "AnswerDisambigF1Correctness":
                print("Evaluating AnswerDisambigF1Correctness...")
                metric = self.get_metric(m)
                ds = dataset.add_column("gt_answers", dataset["long_answers"])
            score, ds = metric.compute(ds, self.batch_size)
            self.results[metric.name] = score
            scores[metric.name] = ds[metric.name]
        scores = [{k:v[i] for k,v in scores.items()} for i in range(len(dataset))]
        dataset = dataset.add_column("scores", scores)

        print("Calculating DR score...")
        def dr_score(d:dict):
            d['scores']['DR_score'] = math.sqrt(d['scores']["answer_disambig_f1"] * d['scores']["answer_rouge_correctness"])
            return d
        dataset = dataset.map(dr_score)
        self.results["DR_score"] = math.sqrt(self.results["answer_disambig_f1"] * self.results["answer_rouge_correctness"])

        print("Evaluation finished.")
        print(f"Results: {self.results}")
        self.dataset = dataset
        return dataset
    
    def save_result(self, dataset_name:str = "result_dataset", result_name:str = "results") -> None:
        """Save the result to files."""
        with open(os.path.join(self.output_dir, result_name)+".json", "w") as f:
            json.dump(self.results, f, indent=4)
        self.dataset.to_json(os.path.join(self.output_dir, dataset_name)+".jsonl")
        print(f"Results saved to {self.output_dir}/results.json and {self.output_dir}/result_datasets.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="benchmarks/ASQA/output")
    args = parser.parse_args()

    benchmark = ASQABenchmark(output_dir=args.output_dir)

    dataset = benchmark.load_data(path="json", data_files=os.path.join(args.output_dir, "dataset.jsonl"), split="train")

    dataset = benchmark.evaluate(dataset)

    benchmark.save_result()

