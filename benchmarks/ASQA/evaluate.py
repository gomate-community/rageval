from datasets import Dataset, load_dataset
import pandas as pd
import json
import re
import os
from typing import List, Dict, Tuple, Optional
import logging

import argparse

from rageval.metrics import (AnswerRougeCorrectness, AnswerEMCorrectness,  AnswerDisambigF1Correctness)

logger = logging.getLogger(__name__)

def evaluate(dataset: Dataset) -> Tuple[Dict[str, float], Dataset]:
    metrics = [AnswerDisambigF1Correctness(), 
               AnswerRougeCorrectness(), 
               AnswerEMCorrectness()]
    for metric in metrics:
        score, dataset = metric.compute(dataset, 1)
        print(f"{metric.name}: {score}")

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="benchmarks/ASQA/output")
    args = parser.parse_args()

    print("\nLoad ASQA dataset...")
    jsonObj = pd.read_json(path_or_buf=f"{args.output_dir}/dataset.json", lines=True)
    dataset = Dataset.from_pandas(jsonObj)
    dataset = dataset.map(lambda example: {'gt_answers': [ann['long_answers'] for ann in example['annotations']]})

    print("Start evaluate...")
    dataset = evaluate(dataset)

    dataset.to_json(f"{args.output_dir}/result_datasets.json")



