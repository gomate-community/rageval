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

    def _recode_gt_supporting_facts(self, data: object) -> object:
        """To calculate f1 recode gt_sent_ids by linking title and index"""
        recode_answers = []
        for title, sent_id in zip(data['supporting_facts']['title'], data['supporting_facts']['sent_id']):
            recode = title.replace(" ","")+ str(sent_id)
            recode_answers.append(recode)
        recode_answers = [' '.join(recode_answers)]
        data["gt_sent_ids"] = recode_answers
        return data

    def _evaluate(self) -> Tuple[Dict[Any, Any], Dataset]:
        """Evaluate the dataset and return the dataset with scores.

        For the HotPotQA dataset(Distractor Setting), we evaluate models by using the `short_answer` and `supporting_answer`.

        For the HotPotQA dataset(Fullwiki Setting), we evaluate models by using the `response`.

        In Distractor Setting,we use the `answer` as the `gt_answers` to evaluate the string Exact Match correctness and the `supporting_facts` to make "gt_sent_ids" to evaluate the F1.

        In Fullwiki Setting,we use the `answer` as the `gt_answers` to evaluate the string Exact Match correctness.
        """

        self.metrics = [AnswerEMCorrectness(ignore_case=True),
                        AnswerF1Correctness()
                        ]
        if (("supporting_answer" in self.dataset.column_names) and "short_answer" in self.dataset.column_names):
            self.dataset = self.dataset.map(self._recode_gt_supporting_facts)
            self.dataset = self.dataset.map(lambda exmaple: {"answer": [[exmaple['answer']]]})
            ground_truths = {
                "answer_f1": ("supporting_answer", "gt_sent_ids"),
                "answer_exact_match": ("short_answer", "answer")
            }
        else:
            self.dataset = self.dataset.map(lambda exmaple: {"answer": [[exmaple['answer']]]})
            ground_truths = {
                "answer_exact_match": ("response", "answer")
            }

        results = {}

        for metric in self.metrics:
            if metric.name in ground_truths:
                print(f"Calculating {metric.name}...")

                if metric.name in self.dataset.column_names:
                    self.dataset = self.dataset.remove_columns(metric.name)

                an, gtan = ground_truths[metric.name]
                self.dataset = self.dataset.rename_column(an, "answers")
                self.dataset = self.dataset.rename_column(gtan, "gt_answers")

                results[metric.name], self.dataset = metric.compute(self.dataset, self.batch_size)

                self.dataset = self.dataset.rename_column("answers", an)
                self.dataset = self.dataset.rename_column("gt_answers", gtan)
        self.dataset = self.dataset.map(lambda example: {"answer": example['answer'][0][0]})
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
        results = benchmark.evaluate(
            path='json',
            data_files={"test": data_file},
            split="test"
        )
        print(f"Results:\n {results}")
        benchmark.save_results(os.path.join(args.output_dir, 'results', f"{args.local_file[:-5]}_{date}.jsonl"))
        benchmark.save_dataset(os.path.join(args.output_dir, 'output', f"{args.local_file[:-5]}_{date}.jsonl"))
    else:
        results = benchmark.evaluate(path='golaxy/rag-bench', name='hotpot_qa', split=args.remote_split)
        print(f"Results:\n {results}")
        benchmark.save_results(os.path.join(args.output_dir, 'results', f"{args.remote_split}_{date}.jsonl"))
        benchmark.save_dataset(os.path.join(args.output_dir, 'output', f"{args.remote_split}_{date}.jsonl"))
