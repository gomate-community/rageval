import argparse
import json
import os
import random
import pandas as pd
from datasets import Dataset
from typing import List, Any, Tuple, Dict

from rageval.models.openai import OpenAILLM
from prompt import (SYNTHETIC_QUERY_FEW_SHOT, SYNTHETIC_QUERY_SYSTEM, SYNTHETIC_QUERY_USER, SYNTHETIC_ANSWER_SYSTEM, SYNTHETIC_ANSWER_USER)

usage = {
    "prompt_tokens":0,
    "completion_tokens": 0,
    "total_tokens": 0
}

def load_corpus(corpus_dir):
    with open(f"{corpus_dir}/corpus.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    df = pd.DataFrame(docs)
    df.drop_duplicates(inplace=True)
    dataset = Dataset.from_dict({'document':df[0].apply(lambda x: x.strip())})

    with open(f"{corpus_dir}/few_shot_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    cases = random.sample(cases, 3)
    return dataset, cases

def generate_responses(engine: OpenAILLM, user_prompts: List[List[str]], system_prompt: List[str]) -> Tuple[List[str], dict]:
    '''Generate responses from the OpenAILLM model.'''
    global usage
    responses = engine.batch_generate(user_prompts, system_roles=system_prompt * len(user_prompts))
    response_texts = [r.generations[0][0].text for r in responses]

    costs = [r.llm_output["token_usage"] for r in responses]
    usage["prompt_tokens"] += sum([c["prompt_tokens"] for c in costs])
    usage["completion_tokens"] += sum([c["completion_tokens"] for c in costs])
    usage["total_tokens"] += sum([c["total_tokens"] for c in costs])

    return response_texts

def generate_questions(engine: OpenAILLM, dataset: Dataset, cases) -> Dataset:
    system_prompt = [SYNTHETIC_QUERY_SYSTEM]
    few_shot_cases = ""
    for i in range(len(cases)):
        few_shot_cases += SYNTHETIC_QUERY_FEW_SHOT.format(
            document=cases[i]["document"], question=cases[i]["Query"])
    user_prompts = [[SYNTHETIC_QUERY_USER.format(
        few_shot_cases=few_shot_cases, document=d['document'])] for d in dataset]
    
    questions = generate_responses(engine, user_prompts, system_prompt)
    return dataset.add_column("question", questions)

def generate_answers(engine: OpenAILLM, dataset: Dataset) -> Dataset:
    system_prompt = [SYNTHETIC_ANSWER_SYSTEM]
    user_prompts = [[SYNTHETIC_ANSWER_USER.format(
        question=d['question'], document=d['document']) + "\n"] for d in dataset]
    
    answers = generate_responses(engine, user_prompts, system_prompt)
    return dataset.add_column("answer", answers)

def validate_question_with_answer(dataset: Dataset) -> Dataset:
    def check_generated_answer(answer: str):
        problematic_phrases = ["I don't know", "i don't know"]
        for phrase in problematic_phrases:
            if phrase in answer.lower():
                return False
        return True
    # dataset.filter(lambda x: not check_generated_answer(x["answer"])).to_json(f"{args.output_dir}/filtered_dataset.json")
    return dataset.filter(lambda x: check_generated_answer(x["answer"]))

def calculate_api_cost(model_name: str):
    # More detail for api prices: https://openai.com/pricing/
    global usage
    # $ / 1k tokens:
    mapping = {
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "gpt-3.5-turbo-16k": (0.003, 0.004), # outdated
        "gpt-4": (0.03, 0.06),
        "gpt-4-32k": (0.06, 0.12),
    }

    intokens = usage["prompt_tokens"]
    outtokens = usage["completion_tokens"]

    if model_name in mapping.keys():
        print(f"Total tokens: {usage['total_tokens']}")
        print(f"Total cost: {mapping[model_name][0] * intokens / 1000 + mapping[model_name][1] * outtokens / 1000}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="benchmark/auto/corpus")
    parser.add_argument("--output_dir", type=str, default="benchmark/auto/output")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.api_key
    engine = OpenAILLM(args.model, "OPENAI_API_KEY")

    print(f"\nLoad corpus from {args.corpus_dir}")
    dataset, cases = load_corpus(args.corpus_dir)

    print("Start generate questions...")
    dataset = generate_questions(engine, dataset, cases) 

    print("Start generate answers...")
    dataset = generate_answers(engine, dataset) 

    print("Validate questions...")
    dataset = validate_question_with_answer(dataset)

    calculate_api_cost(args.model)

    dataset.to_json(f"{args.output_dir}/dataset.json")
    print(f"\nFinish generate dataset. Dataset saved as {args.output_dir}/dataset.json")
