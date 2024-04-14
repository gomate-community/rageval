import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rageval.models.openai import OpenAILLM


def make_doc_prompt(doc, doc_id, doc_prompt, method):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    if method == "vanilla":
        text = doc['text']
    elif method == "summary":
        text = doc["summary"]
    elif method == "snippet":
        text = doc["extraction"]
    else:
        raise ValueError("Don't support such method.")
    return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))


def make_demo(item, prompt, ndoc, doc_prompt, instruction, method, test=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    doc_texts = []
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "")  # if there is no doc we also delete the empty line
        else:
            doc_list = item["docs"][:ndoc]
            for doc_id, doc in enumerate(doc_list):
                doc_texts.append(make_doc_prompt(doc, doc_id, doc_prompt, method=method))
            text = "".join(doc_texts)
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip()  # remove any space or \n

    return prompt, doc_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="gtr")
    parser.add_argument("--method", type=str, default="vanilla")
    parser.add_argument("--ndoc", type=int, default=5)
    parser.add_argument("--shot", type=int, default=2)
    args = parser.parse_args()

    print("-" * 10 + "Loading dataset" + "-" * 10)

    if args.dataset == "gtr":
        eval_data = json.load(
            open(args.cache_path + "/datasets/ALCE-data/asqa_eval_gtr_top100.json", "r")
        )
    elif args.dataset == "oracle":
        eval_data = json.load(
            open(args.cache_path + "/datasets/ALCE-data/asqa_eval_gtr_top100_reranked_oracle.json", "r")
        )
    else:
        raise ValueError("Don't support such dataset.")

    print("-" * 10 + "Finish loading dataset" + "-" * 10)

    print("-" * 10 + "Generating prompts" + "-" * 10)

    eval_prompt = json.load(
        open("benchmarks/ALCE/ASQA/prompts/asqa_prompt.json", "r")
    )

    head_prompt = ""
    for demo_id in range(args.shot):
        demo_item = eval_prompt["demos"][demo_id]
        prompt, _ = make_demo(
            demo_item,
            prompt=eval_prompt["demo_prompt"],
            ndoc=args.ndoc,
            doc_prompt=eval_prompt["doc_prompt"],
            instruction=eval_prompt["instruction"],
            method=args.method
        )
        head_prompt += prompt
        head_prompt += eval_prompt["demo_sep"]

    for idx, eval_item in enumerate(tqdm(eval_data)):
        prompt, doc_texts = make_demo(
            eval_item,
            prompt=eval_prompt["demo_prompt"],
            ndoc=args.ndoc,
            doc_prompt=eval_prompt["doc_prompt"],
            instruction=eval_prompt["instruction"],
            method=args.method,
            test=True
        )
        eval_data[idx]['prompt'] = head_prompt + prompt
        eval_data[idx]['contexts'] = doc_texts
        eval_data[idx]['docs'] = eval_item["docs"][:args.ndoc]

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
        tokenizer = AutoTokenizer.from_pretrained(args.cache_path+"/models/"+args.model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.cache_path+"/models/"+args.model,
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

    file_name = f"asqa-{args.dataset}-{model_name}-{args.method}-shot{args.shot}-ndoc{args.ndoc}"
    file_name = file_name.replace("-", "_")
    result_path = args.cache_path + "/results/" + file_name + ".json"
    json.dump(eval_data, open(result_path, "w"), indent=4)

    print(f"\nResult file saved as {result_path}")
