# HotPotQA BENCHMARK

## 1. Description

This benchmark is designed to evaluate the performance of the [HotPotQA dataset](https://huggingface.co/datasets/hotpot_qa). 

To generate RAG results, we usE `gpt-3.5-turbo-0125` with no retrieval settings, as implemented in [generate.py](generate.py).

## 2. Dataset

The HotPotQA dataset is a question-answering dataset that contains questions which require finding and reasoning over multiple supporting documents to answer and  diverse and not constrained to any pre-existing knowledge bases or knowledge schemas. The benchmark evaluates the correctness of the answers in the dataset.

The sturcture of the dataset is as follows:
```json
{
    "answer": "yes",
    "context": {
        "title": ["Ed Wood (film)","Scott Derrickson","Woodson, Arkansas","Tyler Bates","Ed Wood","Deliver Us from Evil (2014 film)","Adam Collis","Sinister (film)","Conrad Brooks","Doctor Strange (2016 film)"],
        "sentences": [["Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood."," The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau."," Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast."],["Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer."," He lives in Los Angeles, California."," He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\""],["Woodson is a census-designated place (CDP) in Pulaski County, Arkansas, in the United States."," Its population was 403 at the 2010 census."," It is part of the Little Rock\u2013North Little Rock\u2013Conway Metropolitan Statistical Area."," Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century."," Woodson is adjacent to the Wood Plantation, the largest of the plantations own by Ed Wood Sr."],["Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games."," Much of his work is in the action and horror film genres, with films like \"Dawn of the Dead, 300, Sucker Punch,\" and \"John Wick.\""," He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn."," With Gunn, he has scored every one of the director's films; including \"Guardians of the Galaxy\", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel."," In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums \"The Pale Emperor\" and \"Heaven Upside Down\"."],["Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director."],["Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer."," The film is officially based on a 2001 non-fiction book entitled \"Beware the Night\" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was \"inspired by actual accounts\"."," The film stars Eric Bana, \u00c9dgar Ram\u00edrez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014."],["Adam Collis is an American filmmaker and actor."," He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010."," He also studied cinema at the University of Southern California from 1991 to 1997."," Collis first work was the assistant director for the Scott Derrickson's short \"Love in the Ruins\" (1995)."," In 1998, he played \"Crankshaft\" in Eric Koyanagi's \"Hundred Percent\"."],["Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill."," It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger."],["Conrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor."," He moved to Hollywood, California in 1948 to pursue a career in acting."," He got his start in movies appearing in Ed Wood films such as \"Plan 9 from Outer Space\", \"Glen or Glenda\", and \"Jail Bait.\""," He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor."," He also has since gone on to write, produce and direct several films."],["Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures."," It is the fourteenth film of the Marvel Cinematic Universe (MCU)."," The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton."," In \"Doctor Strange\", surgeon Strange learns the mystic arts after a career-ending car accident."]]
    },
    "id": "5a8b57f25542995d1e6f1371",
    "level": "hard",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "supporting_facts": {
        "sent_id": [0, 0],
        "title": ["Scott Derrickson","Ed Wood"]
    },
    "type": "comparison"
}
```

For each "question" in the dataset, there is a brief and precise answer in the "answer" and sentence-level supporting facts required for reasoning in the " supporting_facts", the "level" means the difficulty level of the question, the "context" has  additional context relevant to the question ,"id" is an identifier for the question-answer pair and the field "type" indicates a categorization of the question-answer pair.


## 3. Metrics

We adopt the default metrics used in the [HotPotQA paper](https://arxiv.org/abs/1809.09600) as follows:

In distractor setting:
1. [String Exact Match](../../rageval/metrics/_answer_exact_match.py): for each `answer`, check whether it is present in the predictions.
2. [F1](../../rageval/metrics/_answer_f1.py): utilize the model's predictions to extract supporting facts, which are identified by the model. Subsequently, we compute the F1 score by comparing these extracted supporting facts with the 'supporting_facts' provided in the dataset.

In fullwiki setting:
1. [String Exact Match](../../rageval/metrics/_answer_exact_match.py): for each `answer`, check whether it is present in the predictions.

## 4. Usage

### 4.1 Generate examples

Replace api_key to your OpenAI api key and subset in `run_generate.sh` then run it to generate `gpt-3.5-turbo` response. The command is as follows:

For distractor setting:
```bash
python3 benchmarks/HOTPOTQA/generate.py \
        --subset "distractor" \
        --num_documents 10 \
        --max_num_examples 500 \
        --max_length 4096 \
        --output_path "benchmarks/HOTPOT/output" \
        --cache_path $cache_dir\
        --model "gpt-3.5-turbo" \
        --api_key "YOUR_API_KEY"
```
For fullwiki setting:
```bash
python3 benchmarks/HOTPOTQA/generate.py \
        --subset "fullwiki" \
        --num_documents 10 \
        --max_num_examples 500 \
        --max_length 4096 \
        --output_path "benchmarks/HOTPOT/output" \
        --cache_path $cache_dir\
        --model "gpt-3.5-turbo" \
        --api_key "YOUR_API_KEY"
```
Arguements:

- `--subset`: The maximum number of examples used in generate answers.
    - `"distractor"`:In distractor setting,to challenge the model to find the true supporting facts in the presence of noise,we will load dataset from [huggingface](https://huggingface.co/datasets/hotpotqa/hotpot_qa) which "context" have 8 paragraphs from Wikipedia as distractors and the 2 gold paragraphs.
    - `"fullwiki"`: In fullwiki setting,we will load dataset from `cache_path` which "context" is paragraphs obtained using retrieval system.
- `--num_documents`:The parameter will use in the fullwiki setting to select top-n documents.
- `--max_num_examples`: The maximum number of examples used in generate answers.
- `--max_length`: The maximum number of tokens that can be generated for answering questions.
- `--output_path`: Directory that the generated answers will be saved.
- `--cache_path`: Directory save dataset  which "context" is paragraphs obtained using your retrieval system.We will load the json from "./datasets/hotpot_dev_fullwiki.json".
- `--model`: The OpenAI GPT model to use, e.g., gpt-3.5-turbo. And result file will be named as `f"{model}.jsonl"`.
- `--api_key`: Your OpenAI API key.

### 4.2 Evaluation

1. Prepare RAG responses. By default, `hotpotqa_benchmark.py` will download the results of the `gpt_3.5_turbo` model from [our huggingface dataset](https://huggingface.co/datasets/golaxy/rag-bench) and evaluate them. In distractor setting,if you wish to evaluate your own results, you can simply attach your predictions as `short_answer` and `supporting_answer`  to each example in the original HotPotQA dataset, similar to what we did in [the file](https://huggingface.co/datasets/golaxy/rag-bench/viewer/hotpot_qa/gpt_3.5-_turbo).In fullwiki setting,if you wish to evaluate your own results, you can simply attach your predictions as `response` to each example in the original HotPotQA dataset.

2. Evaluate the responses by replacing local_file running `run.sh`. This script initiates dataset evaluation, with all output files being saved in the output directory. The results will be stored in the `result/` path, which contains scores for all metrics. The command is as follows:

    You can download pre-generated result file from HuggingFace for evaluation.

    ```bash
    python3 benchmarks/HOTPOTQA/hotpot_qa_benchmark.py \
            --output_dir ".rageval/benchmark" \
            --remote_split "gpt_3.5_turbo" 
    ```
    You can also specify locally saved result file for evaluation.
    ```bash
    python3 benchmarks/HOTPOTQA/hotpot_qa_benchmark.py \
            --output_dir ".rageval/benchmark" \
            --local_file  "YOUR_LOCAL_FILE"
    ```
    Arguements:

    - `--output_dir`: Output directory to save results.
    - `--remote_split`: Split dataset from [our huggingface dataset](https://huggingface.co/datasets/golaxy/rag-bench) to evaluate.
    - `--local_file`: Specify locally saved result file to evaluate.


## 5. Performance

Here are results of different models.

In distractor setting:

| Model | STR-EM | F1 |
|:---:|:---:|:---:|
| [gpt-3.5-turbo](https://huggingface.co/datasets/golaxy/rag-bench/viewer/hotpot_qa/gpt_3.5-_turbo) | 44.8 | 47.8 |

In fullwiki setting:

| Model |                          IR Model                          | Top-n Documents | STR-EM |
|:----|:----------------------------------------------------------:|:---------------:|:------:|
| [gpt-3.5-turbo](https://huggingface.co/datasets/golaxy/rag-bench/viewer/hotpot_qa/gpt_3.5-_turbo) | [TF-IDF](https://aclanthology.org/P17-1171/)<sup>[1]</sup> |       10        |  38.4  |
| [gpt-3.5-turbo](https://huggingface.co/datasets/golaxy/rag-bench/viewer/hotpot_qa/gpt_3.5-_turbo) |                      - <sup>[2]</sup>                      |        2        |  62.6  |
| [gpt-3.5-turbo](https://huggingface.co/datasets/golaxy/rag-bench/viewer/hotpot_qa/gpt_3.5-_turbo) |                      no-retrieval                       |        0        |  31.0  |

[1] From [Hotpotqa](https://github.com/hotpotqa/hotpot/blob/master/README.md)

[2] The retrieval results only have 2 gold paragraphs.

## 6. Citations

``` bibtex
@inproceedings{yang2018hotpotqa,
  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year={2018}
}
```
