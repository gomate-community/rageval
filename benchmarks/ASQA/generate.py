from datasets import Dataset, load_dataset
import re
from rageval.models import OpenAILLM
import os
import logging
import argparse

from prompts import (FEW_SHOT_EXAMPLES, PROMPT)

def extract_key_information(pred: str) -> str:
    '''Extract key information from the response.'''
    pattern = r"(?:1\.|\(1\)).*?((?:1\.|\(1\)).*)" # find the second list starting with 1. or (1)
    pred = pred.strip().strip()
    matches = re.findall(pattern, pred, re.DOTALL)
    if matches:
        pred = matches[0]
    else:
        print(f"Cannot extract key information from the response: {pred}")
        return pred
    pred = re.sub(r'\(\d+\)\s', '', pred) # remove the index numbers
    return pred

def generate_answers(engine: OpenAILLM, dataset: Dataset) -> Dataset:
    prompts = [
        PROMPT.format(few_shot_examples=FEW_SHOT_EXAMPLES, 
                      question=data['ambiguous_question'])
        for data in dataset
    ]
    responses = engine.batch_generate(prompts)
    response_texts = [r.generations[0][0].text for r in responses]
    answers = [extract_key_information(response) for response in response_texts]
    dataset = dataset.add_column("responses", response_texts)
    return dataset.add_column("answers", answers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_examples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="benchmarks/ASQA/output")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-instruct")
    parser.add_argument("--api_key", type=str, default=None)

    args = parser.parse_args()

    print("\nLoad ASQA dataset...")
    dataset = load_dataset("din0s/asqa")
    dataset = dataset['dev'].select(range(args.max_num_examples))

    print("Init ASQA dataset...")
    os.environ['OPENAI_API_KEY'] = args.api_key
    engine = OpenAILLM(args.model, 
                         _api_key_env_var = 'OPENAI_API_KEY', 
                         max_tokens=args.max_new_tokens)

    print("Start generate answers...")
    dataset = generate_answers(engine, dataset)

    file_path = os.path.join(args.output_path, f"{args.model}.jsonl")
    dataset.to_json(file_path)
    print(f"\nFinish generate dataset. Dataset saved as {file_path}")

    engine.calculate_api_cost()
