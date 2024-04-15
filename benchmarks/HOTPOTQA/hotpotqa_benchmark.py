import argparse
import os
import time
from typing import Dict, Tuple, Any

from datasets import Dataset

from benchmarks import BaseBenchmark
from rageval.metrics import (AnswerEMCorrectness, AnswerF1Correctness)


class HOTPOTQABenchmark(BaseBenchmark):
    """Benchmark for HotPotQA dataset.

    HotPotQA is a new dataset with 113k Wikipedia-based question-answer pairs with four key features: (1) the questions require finding and reasoning over multiple supporting documents to answer; (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas; (3) we provide sentence-level supporting facts required for reasoning, allowingQA systems to reason with strong supervision and explain the predictions; (4) we offer a new type of factoid comparison questions to test QA systems’ ability to extract relevant facts and perform necessary comparison.

    """

    name = "hotpot_qa_benchmark"

    def __init__(self) -> None:
        """Initialization."""
        super().__init__()

    def _evaluate(self, ) -> Tuple[Dict[Any, Any], Dataset]:
        """Evaluate the dataset and return the dataset with scores.

        For the HotPotQA dataset, the `short_answers` and `long_answers` which we need to extract.


        We use the `answer` as the `gt_answers` to evaluate the string Exact Match correctness and the `gt_supporting_facts` to evaluate the F1.
        """
        self.metrics = [AnswerEMCorrectness(ignore_case=False),
                        AnswerF1Correctness()
                        ]
        ground_truths = {
            "answer_f1": ("supporting_answer", "gt_supporting_facts"),
            "answer_exact_match": ("short_answer", "answer")
        }

        results = {}

        for metric in self.metrics:
            if metric.name in ground_truths:
                print(f"Calculating {metric.name}...")
                an, gtan = ground_truths[metric.name]
                self.dataset = self.dataset.rename_column(an, "answers")
                self.dataset = self.dataset.rename_column(gtan, "gt_answers")

                results[metric.name], self.dataset = metric.compute(self.dataset, self.batch_size)


                self.dataset = self.dataset.rename_column("answers", an)
                self.dataset = self.dataset.rename_column("gt_answers", gtan)
        return results, self.dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="benchmarks/HOTPOTQA")
    parser.add_argument("--remote_split", type=str, default="gpt_3.5_turbo")
    parser.add_argument("--local_file", type=str, default=None)

    args = parser.parse_args()
    date = time.strftime("%Y%m%d", time.localtime())

    benchmark = HOTPOTQABenchmark()
    if args.local_file:
        data_file = os.path.join(args.output_dir, 'output', args.local_file)
        print(data_file)
        print(data_file)
        results = benchmark.evaluate(
            path='json',
            data_files={"test": data_file},
            split="test"
        )
        print(f"Results:\n {results}")
        benchmark.save_results(os.path.join(args.output_dir, 'results', f"{args.local_file[:-5]}_{date}.jsonl"))
    else:
        results = benchmark.evaluate(path='golaxy/rag-bench', name='hotpot_qa', split=args.remote_split)
        print(f"Results:\n {results}")
        benchmark.save_results(os.path.join(args.output_dir, 'results', f"{args.remote_split}_{date}.jsonl"))
