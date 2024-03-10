import json
import os
import time

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import rageval as rl
from rageval.models.openai import OpenAILLM
from utils import make_demo, create_eli5_eval_dataset


class ELI5:

    def __init__(self, args):
        self.args = args

        print("-" * 10 + "Loading model" + "-" * 10)
        if "gpt-3.5-turbo" in self.args.model:
            os.environ["OPENAI_API_KEY"] = self.args.api_key
            self.model = OpenAILLM(
                model=self.args.model,
                _api_key_env_var="OPENAI_API_KEY",
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model,
                device_map='auto'
            )
        print("-" * 10 + "Finish loading model" + "-" * 10)

        print("-" * 10 + "Loading dataset" + "-" * 10)
        if self.args.dataset == "bm25":
            self.eval_data = json.load(
                open(self.args.cache_path + "/datasets/ALCE-data/eli5_eval_bm25_top100.json", "r")
            )
        elif self.args.dataset == "oracle":
            self.eval_data = json.load(
                open(self.args.cache_path + "/datasets/ALCE-data/eli5_eval_bm25_top100_reranked_oracle.json", "r")
            )
        else:
            raise ValueError("Don't support such dataset.")

        self.prompt = json.load(
            open("benchmarks/ALCE/ELI5/prompts/eli5_prompt.json", "r")
        )
        print("-" * 10 + "Finish loading dataset" + "-" * 10)

        model_name = self.args.model
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        self.name = f"eli5-{self.args.dataset}-{model_name}-{self.args.method}-shot{self.args.shot}-ndoc{self.args.ndoc}"
        self.result_path = ".rageval/results/" + self.name + ".json"

    def generate(self):
        head_prompt = ""
        for demo_id in range(self.args.shot):
            demo_item = self.prompt["demos"][demo_id]
            prompt, _ = make_demo(
                demo_item,
                prompt=self.prompt["demo_prompt"],
                ndoc=self.args.ndoc,
                doc_prompt=self.prompt["doc_prompt"],
                instruction=self.prompt["instruction"],
                method=self.args.method
            )
            head_prompt += prompt
            head_prompt += self.prompt["demo_sep"]

        print("-" * 10 + "Generating prompts" + "-" * 10)
        for idx, eval_item in enumerate(tqdm(self.eval_data)):
            prompt, doc_texts = make_demo(
                eval_item,
                prompt=self.prompt["demo_prompt"],
                ndoc=self.args.ndoc,
                doc_prompt=self.prompt["doc_prompt"],
                instruction=self.prompt["instruction"],
                method=self.args.method,
                test=True
            )
            self.eval_data[idx]['prompt'] = head_prompt + prompt
            self.eval_data[idx]['contexts'] = doc_texts
            self.eval_data[idx]['docs'] = eval_item["docs"][:self.args.ndoc]
        print("-" * 10 + "Finish generating prompts" + "-" * 10)

        for idx, item in enumerate(tqdm(self.eval_data)):
            prompt = item['prompt']
            if "gpt-3.5-turbo" in self.args.model:
                output = self.model.generate(
                    [prompt],
                    "You are a helpful assistant that answers the following questions with proper citations."
                )
                item['output'] = output.generations[0][0].text
            else:
                inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
                stop = ["\n", "Ċ", "ĊĊ", "<0x0A>"]  # In Llama \n is <0x0A>; In OPT \n is Ċ
                stop_token_ids = list(
                    set(
                        [self.tokenizer._convert_token_to_id(stop_token) for stop_token in stop]
                        + [self.model.config.eos_token_id]
                    )
                )
                if "llama" in self.args.model.lower():
                    stop_token_ids.remove(self.tokenizer.unk_token_id)

                generation = self.model.generate(
                    **inputs,
                    max_length=self.args.max_length,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    eos_token_id=stop_token_ids,
                    do_sample=True
                )
                output = self.tokenizer.decode(generation[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
                item['output'] = output

        json.dump(self.eval_data, open(self.result_path, "w"), indent=4)

    def evaluate(self):
        dataset = Dataset.from_generator(create_eli5_eval_dataset, gen_kwargs={"result_path": self.result_path})
        result = {}
        nli_model = rl.models.NLIModel(
            "text2text-generation",
            self.args.cache_path + "/models/t5_xxl_true_nli_mixture",
        )
        if "nli_claim" in self.args.metrics:
            metric = rl.metrics.AnswerNLICorrectness(nli_model=nli_model, decompose_model="nltk")
            score, ds = metric.compute(dataset)
            result["nli_claim"] = 100 * score

        if "citation_recall" in self.args.metrics:
            metric = rl.metrics.AnswerCitationRecall(nli_model=nli_model)
            score, ds = metric.compute(dataset)
            result["citation_recall"] = 100 * score

        if "citation_precision" in self.args.metrics:
            metric = rl.metrics.AnswerCitationPrecision(nli_model=nli_model)
            score, ds = metric.compute(dataset)
            result["citation_precision"] = 100 * score

        date = time.strftime("%Y%m%d", time.localtime())

        json.dump(
            result,
            open(f"benchmarks/ALCE/ELI5/results/{self.name}-{date}.json", "w"),
            indent=4
        )