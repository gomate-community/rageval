# HotPotQA BENCHMARK

## 1. Description

This benchmark is designed to evaluate the performance of the [HotPotQA dataset](https://huggingface.co/datasets/hotpot_qa). 

To generate RAG results, we usE `gpt-3.5-turbo-0125` with no retrieval settings, as implemented in [generate.py](generate.py).

## 2. Dataset

The HotPotQA dataset is a question-answering dataset that contains questions which require finding and reasoning over multiple supporting documents to answer and  diverse and not constrained to any pre-existing knowledge bases or knowledge schemas. The benchmark evaluates the correctness of the answers in the dataset.

The sturcture of the dataset is as follows:
```json
{
    "answer": "This is the answer",
    "context": {
        "sentences": [["Sent 1"], ["Sent 2"]],
        "title": ["Title1", "Title 2"]
    },
    "id": "000001",
    "level": "hard",
    "question": "What is the answer?",
    "supporting_facts": {
        "sent_id": [0, 1, 3],
        "title": ["Title of para 1", "Title of para 2", "Title of para 3"]
    },
    "type": "bridge"
}
```

For each "question" in the dataset, there is a brief and precise answer in the "answer" and sentence-level supporting facts required for reasoning in the " supporting_facts", the "level" means the difficulty level of the question, the "context" has  additional context relevant to the question ,"id" is an identifier for the question-answer pair and the field "type" indicates a categorization of the question-answer pair.


## 3. Metrics

We adopt the default metrics used in the [HotPotQA paper](https://arxiv.org/abs/1809.09600) as follows:

1. [String Exact Match](../../rageval/metrics/_answer_exact_match.py): for each `answer`, check whether it is present in the predictions.
2. [F1](../../rageval/metrics/_answer_f1.py): utilize the model's predictions to extract supporting facts, which are identified by the model. Subsequently, we compute the F1 score by comparing these extracted supporting facts with the 'supporting_facts' provided in the dataset。


## 4. Usage

### 4.1 Generate examples

Replace api_key to your OpenAI api key in `run_generate.sh` then run it to generate `gpt-3.5-turbo` response. The command is as follows:

```bash
python3 benchmarks/HOTPOTQA/generate.py \
        --subset "distractor"\
        --max_num_examples 500 \
        --max_length 4096 \
        --output_path "benchmarks/HOTPOT/output" \
        --model "gpt-3.5-turbo" \
        --api_key "YOUR_API_KEY"
```

Arguements:

- `--max_num_examples`: The maximum number of examples used in generate answers.
- `--max_length`: The maximum number of tokens that can be generated for answering questions.
- `--output_path`: Directory that the generated answers will be saved.
- `--model`: The OpenAI GPT model to use, e.g., gpt-3.5-turbo. And result file will be named as `f"{model}.jsonl"`.
- `--api_key`: Your OpenAI API key.

### 4.2 Evaluation

1. Prepare RAG responses. By default, `hotpotqa_benchmark.py` will download the results of the `gpt_3.5_turbo` model from [our huggingface dataset](https://huggingface.co/datasets/golaxy/rag-bench) and evaluate them. If you wish to evaluate your own results, you can simply attach your predictions as `short_answer` and `supporting_answer`  to each example in the original HotPotQA dataset, similar to what we did in [the file](https://huggingface.co/datasets/golaxy/rag-bench/viewer/hotpot_qa/gpt_3.5-_turbo).

2. Evaluate the responses by running `run.sh`. This script initiates dataset evaluation, with all output files being saved in the output directory. The results will be stored in the `result/` path, which contains scores for all metrics. The command is as follows:

```bash
python3 benchmarks/HOTPOTQA/hotpot_qa_benchmark.py --output_dir "benchmarks/HOTPOTQA" --remote_split "gpt_3.5_turbo"
```

Arguements:

- `--output_dir`: Output directory to save results.
- `--remote_split`: Split dataset from [our huggingface dataset](https://huggingface.co/datasets/golaxy/rag-bench) to evaluate.

## 5. Performance

Here are results of different models.

| Model | STR-EM | F1 |
|:---:|:---:|:---:|
| [gpt-3.5-turbo](https://huggingface.co/datasets/golaxy/rag-bench/viewer/hotpot_qa/gpt_3.5-_turbo) | 54.8 | 39.2 |


## 6. Citations

``` bibtex
@inproceedings{yang2018hotpotqa,
  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year={2018}
}
```