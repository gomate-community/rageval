import argparse
import time
from typing import Dict, Tuple, Any

from datasets import Dataset

import rageval as rl
from benchmarks import BaseBenchmark
from rageval.metrics import AnswerEMCorrectness, AnswerCitationRecall, AnswerCitationPrecision


class ASQABenchmark(BaseBenchmark):

    name = "asqa_benchmark"

    def __init__(self, cache_path) -> None:
        super().__init__()
        nli_model = rl.models.NLIModel(
            "text2text-generation",
            cache_path + "/models/t5_xxl_true_nli_mixture",
        )
        self.metrics = [
            AnswerEMCorrectness(),
            AnswerCitationRecall(nli_model=nli_model),
            AnswerCitationPrecision(nli_model=nli_model)
        ]

    def _evaluate(self) -> Tuple[Dict[Any, Any], Dataset]:
        self.dataset = self.dataset.rename_column("output", "answers")

        gt_short_answers = [
            [
                pair["short_answers"]
                for pair in data["qa_pairs"]
            ]
            for data in self.dataset
        ]
        self.dataset = self.dataset.add_column("gt_answers", gt_short_answers)

        print(self.dataset)

        results = {}
        for metric in self.metrics:
            results[metric.name], self.dataset = metric.compute(self.dataset, self.batch_size)
        return results, self.dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--remote_split", type=str, default=None)
    parser.add_argument("--local_file", type=str, default=None)
    args = parser.parse_args()
    date = time.strftime("%Y%m%d", time.localtime())

    benchmark = ASQABenchmark(cache_path=args.cache_path)
    if args.local_file:
        print(args.cache_path+"/results/"+args.local_file)
        results = benchmark.evaluate(
            path="json",
            data_files={
                "test": args.cache_path+"/results/"+args.local_file
            },
            split="test"
        )
        benchmark.save_results(f"benchmarks/ALCE/ASQA/results/{args.local_file[:-5]}_{date}.json")
    else:
        results = benchmark.evaluate(path="golaxy/rag-bench", name="alce_asqa_gtr", split=args.remote_split)
        benchmark.save_results(f"benchmarks/ALCE/ASQA/results/{args.remote_split}_{date}.json")
