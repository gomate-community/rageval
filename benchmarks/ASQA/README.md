# ASQA BENCHMARK

## 1. Description

This benchmark is designed to evaluate the performance of the [ASQA dataset](https://huggingface.co/datasets/din0s/asqa). For `generate.py`, we followed [FLARE](https://github.com/jzbjyb/FLARE), using `gpt-3.5-turbo-instruct` with no retrieval settings.

## 2. Dataset

The ASQA dataset is a question-answering dataset that contains factoid questions and long-form answers. The benchmark evaluates the correctness of the answers in the dataset.

The sturcture of the dataset is as follows:
```json
{
  "ambiguous_question": Value(dtype="string", id=None),
  "qa_pairs": [{
      "context": Value(dtype="string", id=None),
      "question": Value(dtype="string", id=None),
      "short_answers": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
      "wikipage": Value(dtype="string", id=None)
  }],
  "wikipages": [{
      "title": Value(dtype="string", id=None),
      "url": Value(dtype="string", id=None)
  }],
  "annotations": [{
      "knowledge": [{
          "content": Value(dtype="string", id=None),
          "wikipage": Value(dtype="string", id=None)
      }],
      "long_answer": Value(dtype="string", id=None)
  }],
  "sample_id": Value(dtype="string", id=None)
}
```

For each "ambiguous_question" in the dataset, there is a list of disambiguations named "qa_pairs", each pair consisting of a disambiguated "question", a "short_answer" that indicates the "question", and a "context" that provides evidence for the "short_answer". 

Additionally, the "annotations" contain 2 human annotators' comments, each comment including a "long_answer" that can answer the original "ambiguous_question" and all the disambiguations within it, along with a "knowledge" set supporting the "long_answer".

## 3. Metrics

We adopt the default metrics used in the [ASQA paper](https://aclanthology.org/2022.emnlp-main.566) as follows:

1. RougeL: compare predictions against all `long_answer` provided by human annotators.
2. String Exact Match: for each `short_answer`, check whether it is present in the predictions.
3. DisambigF1: use a RoBERTa-based model to extract entities in the `long_answer` and predictions, then compute the F1 score between two set of entities.
4. DR (Disambiguation-Rouge) Score: the geometric mean of DisambigF1 and RougeL.

If there are multiple ground truth answers in one example, we compute the score between the prediction and every ground truth, and take the *maximum* score as the score of the predictions.

## 4. Usage

### 4.1 Generate examples

Replace api_key to your OpenAI api key in `run_generate.sh` then run it to generate `gpt-3.5-turbo-instruct` response. The command is as follows:

```bash
python3 benchmarks/ASQA/generate.py \
        --max_num_examples 500 \
        --max_new_tokens 256 \
        --output_path "benchmarks/ASQA/output" \
        --model "gpt-3.5-turbo-instruct" \
        --api_key "YOUR_API_KEY" 
```

Arguements:

- `--max_num_examples`: The maximum number of examples used in generate answers.
- `--max_new_tokens`: The maximum number of tokens that can be generated for answering questions.
- `--output_path`: Directory that the generated answers will be saved.
- `--model`: The OpenAI GPT model to use, e.g., gpt-3.5-turbo-instruct. And result file will be named as `f"{model}.jsonl"`.
- `--api_key`: Your OpenAI API key.

### 4.2 Evaluation

1. Prepare RAG responses. By default, `asqa_benchmark.py` will download the results of the `gpt-3.5-turbo-instruct` model from [our huggingface dataset](https://huggingface.co/datasets/golaxy/rag-bench) and evaluate them. If you wish to evaluate your own results, you can simply attach your predictions as `answers` to the end of each example in the original ASQA dataset, similar to what we did in [the file](data\gpt-3.5-turbo-instruct.jsonl).

2. Evaluate the responses. Run `run.sh` to start dataset evaluation. The results will be saved in the output directory, named `results.jsonl`. Additionally, the `result_datasets.jsonl` file is a JSON dump of the detailed results, including scores for every example in the dataset.The command is as follows:

```bash
python3 benchmarks/ASQA/asqa_benchmark.py \
    --output_dir "benchmarks/ASQA/output" \
    --dataset_path "benchmarks/ASQA/data/gpt-3.5-turbo-instruct.jsonl"
```

Arguements:

- `--output_dir`: Output directory to save results.
- `dataset_path`: The dataset including model predictions and ground truths to evaluate.


## 5. Performance

Here are results of different models.

| Model | STR-EM | Rouge-L | Disambig F1 | D-R Score|
|:---:|:---:|:---:|:---:|:---:|
| gpt-3.5-turbo-instruct | 33.8 | 30.2 | 30.7 | 30.5 |
| mistral-7b | 20.6 | 31.1 | 26.6 | 28.7 |
| text-davinci-003 ([Jiang et al.](http://arxiv.org/abs/2305.06983)) | 33.8 | 33.3 | 24.2 | 28.4 |
| PALM-540B ([Amplayo et al.](https://aclanthology.org/2023.acl-long.444))| - | 34.5 | 25.3 | 29.6 |

## 6. Citations

``` bibtex
@article{stelmakh2022asqa,
    title={ASQA: Factoid questions meet long-form answers},
    author={Stelmakh, Ivan and Luan, Yi and Dhingra, Bhuwan   and Chang, Ming-Wei},
    journal={arXiv preprint arXiv:2204.06092},
    year={2022}
}

@article{jiang2023flare,
    title={Active Retrieval Augmented Generation}, 
    author={Zhengbao Jiang and Frank F. Xu and Luyu Gao and Zhiqing Sun and Qian Liu and Jane Dwivedi-Yu and Yiming Yang and Jamie Callan and Graham Neubig},
    year={2023},
    eprint={2305.06983},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@inproceedings{amplayo-etal-2023-query,
    title = "Query Refinement Prompts for Closed-Book Long-Form {QA}",
    author = "Amplayo, Reinald Kim  and
      Webster, Kellie  and
      Collins, Michael  and
      Das, Dipanjan  and
      Narayan, Shashi",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.444",
    doi = "10.18653/v1/2023.acl-long.444",
    pages = "7997--8012",
}
```
