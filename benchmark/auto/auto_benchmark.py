from rageval.models.openai import OpenAILLM
from prompt import (SYNTHETIC_QUERY_FEW_SHOT, SYNTHETIC_QUERY_SYSTEM, SYNTHETIC_QUERY_USER)

import json
import os
import random
from datasets import Dataset

# load corpus and few-shot cases
with open("benchmark/auto/corpus/corpus.json", "r", encoding="utf-8") as f:
    docs = json.load(f)
dataset = Dataset.from_dict({"document": docs})

with open("benchmark/auto/corpus/few_shot_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)
cases = random.sample(cases, 3)

# generate prompts
system_prompt = [SYNTHETIC_QUERY_SYSTEM]
few_shot_cases = ""
for i in range(len(cases)):
    few_shot_cases += SYNTHETIC_QUERY_FEW_SHOT.format(document=cases[i]["document"], question=cases[i]["Query"]) + "\n"
user_prompts = []
for i in range(len(docs)):
    user_prompts.append([SYNTHETIC_QUERY_USER.format(few_shot_cases=few_shot_cases, document=docs[i]) + "\n"])

# generate responses
os.getenv("OPENAI_API_KEY")
engine = OpenAILLM("gpt-3.5-turbo", "OPENAI_API_KEY")
responses = engine.batch_generate(user_prompts, system_roles=system_prompt * len(user_prompts))
for i in range(len(responses)):
    print(responses[i])
    print("\n")
