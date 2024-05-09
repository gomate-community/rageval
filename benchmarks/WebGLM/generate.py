import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rageval.models.openai import OpenAILLM


PROMPT = "Answer the question based on the following references with citations. Use a mark for each helpful reference you cited, such as [1]. If there are multiple citations at one position, please use a format like [1][2][3]. If a reference is useless, do not cite it."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    print("-" * 10 + "Loading dataset" + "-" * 10)

    eval_data = []
    with open(args.cache_path + "/datasets/webglm-test.jsonl", "r") as read_file:
        for line in tqdm(read_file):
            eval_data.append(json.loads(line))

    print("-" * 10 + "Finish loading dataset" + "-" * 10)

    print("-" * 10 + "Generating prompts" + "-" * 10)

    for idx, item in enumerate(tqdm(eval_data)):
        prompt = PROMPT + '\n'
        for ix, ref in enumerate(item["references"]):
            prompt += f'Reference [{ix + 1}]: {ref}\n'
        prompt += f'Question: {item["question"]}\nAnswer: '
        eval_data[idx]["prompt"] = prompt

    print("-" * 10 + "Finish generating prompts" + "-" * 10)

    print("-" * 10 + "Loading model" + "-" * 10)

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

    print("-" * 10 + "Finish loading model" + "-" * 10)

    print("-" * 10 + "Predict" + "-" * 10)

    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']
        if "gpt" in model_name:
            output = model.generate(
                inputs=[prompt],
                system_role="You are a helpful assistant that answers the following questions with proper citations."
            )
            item['output'] = output.generations[0][0].text
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
            item['output'] = output

    print("-" * 10 + "Finish predicting" + "-" * 10)

    file_name = f"webglm-{model_name}"
    file_name = file_name.replace("-", "_")
    result_path = args.cache_path + "/results/" + file_name + ".json"
    json.dump(eval_data, open(result_path, "w"), indent=4)

    print(f"\nResult file saved as {result_path}")
