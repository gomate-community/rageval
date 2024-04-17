import argparse
import os
import re

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rageval.models import OpenAILLM
from prompts import (FEW_SHOT_EXAMPLES, PROMPT)

from typing import Tuple


def process_response(response: str) -> Tuple[str, str]:
    answer, supporting = "", []
    response = response.strip()[1:-1].split('\n')
    if response:
        answer = response[0]
        if len(response) != 1:
            len_support = [len(s) for s in data['context']['sentences']]
            for sup in response[1:]:
                t = sup.split(':')
                if t and len(t) != 1:
                    title = t[0]
                    if title in data['context']['title']:
                        idx = data['context']['title'].index(title)
                        supporting += [str(1 + sum(len_support[:idx]) + int(re.sub(r'\D', '', v))) for v in
                                       t[1].replace('[',"").replace(']',"").split(',') if re.sub(r'\D', '', v)]
    if supporting:
        supporting = " ".join(supporting)
    else:
        supporting = ""
    return answer, supporting


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--subset", type=str, default="distractor")
    parser.add_argument("--max_num_examples", type=int, default=500)

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--output_path", type=str,
                        default="benchmarks/HOTPOTQA/output")

    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")

    parser.add_argument("--api_key", type=str, default=None)

    args = parser.parse_args()

    print("\nLoad HotPotQA dataset...")
    dataset = load_dataset('hotpot_qa', args.subset)
    eval_data = dataset['validation'].select(range(args.max_num_examples))

    print("Init HotPotQA dataset...")

    print("\nLoad model...")
    model_name = args.model.split("/")[-1]
    if "gpt" in model_name:
        os.environ["OPENAI_API_KEY"] = args.api_key
        model = OpenAILLM(
            model=model_name,
            _api_key_env_var="OPENAI_API_KEY",
            max_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.cache_path + "/models/" + args.model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.cache_path + "/models/" + args.model,
            device_map='auto'
        )

    print("Start generate answers...")

    model_name = args.model.split("/")[-1]

    outputs, answers, supporting_facts = [], [], []
    for data in eval_data:

        re_context = [{t: s} for t, s in zip(data['context']['title'], data['context']['sentences'])]

        prompt = PROMPT.format(few_shot_examples=FEW_SHOT_EXAMPLES,
                               Q=data['question'], Tp=data['type'], C=re_context)

        if "gpt" in model_name:
            output = model.generate(
                inputs=[prompt],
                system_role="You're a helpful assistant that provides concise and accurate answers to the following questions."
            )
            outputs.append(output.generations[0][0].text)
            answer, supporting = process_response(output.generations[0][0].text)
            answers.append(answer)
            supporting_facts.append(supporting)
        else:
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            stop = ["\n", "Ċ", "ĊĊ", "<0x0A>"]  # In Llama \n is <0x0A>; In OPT \n is Ċ
            stop_token_ids = list(
                set(
                    [tokenizer._convert_token_to_id(stop_token) for stop_token in stop]
                    + [model.config.eos_token_id]
                )
            )
            if "llama" in model_name.lower():
                stop_token_ids.remove(tokenizer.unk_token_id)

            generation = model.generate(
                **inputs,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=stop_token_ids,
                do_sample=True
            )
            output = tokenizer.decode(generation[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            outputs.append(output.generations[0][0].text)
            answer, supporting = process_response(output.generations[0][0].text)
            answers.append(answer)
    eval_data = eval_data.add_column("response", outputs)
    eval_data = eval_data.add_column("short_answer", answers)
    eval_data = eval_data.add_column("supporting_answer", supporting_facts)

    file_path = os.path.join(args.output_path, f"{args.model.replace('-', '_')}.jsonl")
    eval_data.to_json(file_path)
    print(f"\nFinish generate dataset. Dataset saved as {file_path}")
