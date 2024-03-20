from datasets import Dataset, load_dataset
import pandas as pd
import json
import math
from typing import List, Dict, Tuple, Optional
import logging

import argparse

from rageval.metrics import (AnswerRougeCorrectness, AnswerEMCorrectness,  AnswerDisambigF1Correctness)

logger = logging.getLogger(__name__)

def evaluate(dataset: Dataset) -> Tuple[Dict[str, float], Dataset]:
    metrics = [AnswerDisambigF1Correctness(), 
               AnswerRougeCorrectness("rougeL"), 
               AnswerEMCorrectness()]
    results = {}
    for metric in metrics:
        if metric.name == "answer_exact_match":
            dataset = dataset.map(lambda example: {'gt_answers': [ann['short_answers'] for ann in example['qa_pairs']]})
        score, dataset = metric.compute(dataset, 1)
        results[metric.name] = score
    results["DR"] = math.sqrt(results['answer_disambig_f1'] * results['answer_rouge_correctness'])
    dataset = dataset.map(lambda example: {'DR_score': math.sqrt(example['answer_disambig_f1'] * example['answer_rouge_correctness'])})
    return dataset, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="benchmarks/ASQA/output")
    args = parser.parse_args()

    print("\nLoad ASQA dataset...")
    jsonObj = pd.read_json(path_or_buf=f"{args.output_dir}/dataset.jsonl", lines=True)
    dataset = Dataset.from_pandas(jsonObj)
    dataset = dataset.map(lambda example: {'gt_answers': [ann['long_answer'] for ann in example['annotations']]})

    print("Start evaluate...")
    dataset, results = evaluate(dataset)
    print(f"Results: {results}")
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    dataset.to_json(f"{args.output_dir}/result_datasets.json")
