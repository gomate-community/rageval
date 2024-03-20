from datasets import Dataset, load_dataset
import pandas as pd
import json
import re
from rageval.models import OpenAILLM
import os
from typing import List, Dict, Optional
import openai
import logging
from langchain.schema import Generation, LLMResult
import argparse

from prompts import (FEW_SHOT_EXAMPLES, PROMPT)

from rageval.metrics import (AnswerRougeCorrectness, AnswerEMCorrectness,  AnswerDisambigF1Correctness)

logger = logging.getLogger(__name__)

class InstructGPT(OpenAILLM):
    def __init__(self, model: str = "gpt-3.5-turbo-instruct", *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        
    def generate(self, prompt: str, ):
        try:
            response = self.llm.with_options(
                max_retries=self.num_retries,
                timeout=self.timeout).completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens, # defaults to 16 for openai.completions.create
                    n=self.n,
                    temperature=self.temperature,
                    top_p=self.top_p)
            result = self.create_llm_result(response)
            return result
        except openai.APIConnectionError as e:
            logger.info("The server could not be reached")
            logger.info(e.__cause__)  # an underlying Exception, likely raised within httpx.
            raise e
        except openai.RateLimitError as e:
            logger.info("A 429 status code was received; we should back off a bit.")
            raise e
        except openai.APIStatusError as e:
            logger.info("Another non-200-range status code was received")
            logger.info(e.status_code)
            logger.info(e.response)
            raise e
    
    def batch_generate(self, prompts: List[str]) -> List[LLMResult]:
        results = []
        for prompt in prompts:
            try: 
                result = self.generate(prompt)
            except Exception as e:
                result = LLMResult(generations=[[Generation(text="")]], llm_output={})
                print(e)
            results.append(result)
        return results
    
    def create_llm_result(self, response) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        if not isinstance(response, dict):
            response = response.model_dump()

        # token Usage
        token_usage = response.get("usage", {})
        self.usage["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += token_usage.get("completion_tokens", 0)
        self.usage["total_tokens"] += token_usage.get("total_tokens", 0)
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model,
            "system_fingerprint": response.get("system_fingerprint", "")
        }

        choices = response["choices"]
        generations = [
            Generation(
                text=choice["text"],
                generation_info=dict(
                    finish_reason=choice.get("finish_reason"),
                    logprobs=choice.get("logprobs"),
                ),
            )
            for choice in choices
        ]
        return LLMResult(generations=[generations], llm_output=llm_output)

def generate_responses(engine: InstructGPT, prompts: List[str]) -> List[str]:
    '''Generate responses from the OpenAILLM model.'''
    responses = engine.batch_generate(prompts)
    response_texts = [r.generations[0][0].text for r in responses]

    return response_texts

def extract_key_information(pred: str) -> str:
    '''Extract key information from the response.'''
    prefix_to_remove=['The answers to all interpretations are\: (.*)$',
                    'The answer to this interpretation is\: (.*)$',
                    'The answer to this interpretation is (.*)$',
                    'The answer to the first interpretation is: (.*)$']
    for pattern in prefix_to_remove:
        pred = pred.strip().split('\n\n', 1)[0].strip()
        find = re.compile(pattern).search(pred)
        if find:
            pred = find.group(1)
            break
    if find is None:
        logging.warning(f"Cannot extract key information from the response: {pred}")
    return pred

def generete_answers(engine: InstructGPT, dataset: Dataset) -> Dataset:
    prompts = [
        PROMPT.format(few_shot_examples=FEW_SHOT_EXAMPLES, 
                      question=data['ambiguous_question'])
        for data in dataset
    ]
    responses = generate_responses(engine, prompts)

    answers = [extract_key_information(response) for response in responses]

    return dataset.add_column("answers", answers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_examples", type=int, default=5)
    # parser.add_argument("--max_num_fewshots", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="benchmarks/ASQA/output")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-instruct")
    parser.add_argument("--api_key", type=str, default=None)

    args = parser.parse_args()

    dataset = load_dataset("din0s/asqa")
    dataset = dataset['dev'].select(range(args.max_num_examples))
    dataset = dataset.map(lambda example: {'gt_answers': [ann['long_answer'] for ann in example['annotations']]})

    os.environ['OPENAI_API_KEY'] = args.api_key
    engine = InstructGPT(args.model, 
                         _api_key_env_var = 'OPENAI_API_KEY', 
                         max_tokens=args.max_new_tokens)

    dataset = generete_answers(engine, dataset)

    dataset.to_json(f"{args.output_dir}/dataset.json")

    metric = AnswerDisambigF1Correctness()

    score, results = metric.compute(dataset, 5)
    
    print(score,results)
    pass


