# Rageval

Evaluation tools for Retrieval-augmented Generation (RAG) methods.

[![python](https://img.shields.io/badge/Python-3.8.18-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
![workflow status](https://github.com/gomate-community/rageval/actions/workflows/makefile.yml/badge.svg)
[![codecov](https://codecov.io/gh/gomate-community/rageval/graph/badge.svg?token=AH4DNR46HL)](https://codecov.io/gh/gomate-community/rageval)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

Rageval is a tool that helps you evaluate RAG system. The evaluation consists of six sub-tasks, including query rewriting, document ranking, information compression, evidence verify, answer generating, and result validating.

## Definition of tasks and metrics
### 1. [The generate task](./rageval/tasks/_generate.py)
The generate task is to answer the question based on the contexts provided by retrieval modules in RAG. Typically, the context could be extracted/generated text snippets from the compressor, or relevant documents from the re-ranker. Here, we divide metrics used in the generate task into two categories, namely *answer correctness* and *answer groundedness*.

(1) **Answer Correctness**: this category of metrics is to evaluate the correctness by comparing the generated answer with the groundtruth answer. Here are some commonly used metrics:

* [Answer F1 Correctness](./rageval/metrics/_answer_f1.py): is widely used in [the paper (Jiang et al.)](https://arxiv.org/abs/2305.06983), [the paper (Yu et al.)](https://arxiv.org/abs/2311.09210), [the paper (Xu et al.)](https://arxiv.org/abs/2310.04408), and others.
* [Answer NLI Correctness](./rageval/metrics/_answer_claim_recall.py): also known as *claim recall* in [the paper (Tianyu et al.)](https://arxiv.org/abs/2305.14627).
* [Answer EM Correctness](./rageval/metrics/_answer_exact_match.py): also known as *Exact Match* as used in [the paper (Ivan Stelmakh et al.)](https://arxiv.org/abs/2204.06092).
* [Answer Bleu Score](./rageval/metrics/_answer_bleu.py): also known as *Bleu* as used in [the paper (Kishore Papineni et al.)](https://www.aclweb.org/anthology/P02-1040.pdf).
* [Answer Ter Score](./rageval/metrics/_answer_ter.py): also known as *Translation Edit Rate* as used in [the paper (Snover et al.)](https://aclanthology.org/2006.amta-papers.25).
* [Answer chrF Score](./rageval/metrics/_answer_chrf.py): also known as *character n-gram F-score* as used in [the paper (Popovic et al.)](https://aclanthology.org/W15-3049).
* [Answer Disambig-F1](./rageval/metrics/_answer_disambig_f1.py): also known as *Disambig-F1* as used in [the paper (Ivan Stelmakh et al.)](https://arxiv.org/abs/2204.06092) and [the paper (Zhengbao Jiang et al.)](https://arxiv.org/abs/2305.06983).
* [Answer Rouge Correctness](./rageval/metrics/_answer_rouge_correctness.py): also known as *Rouge* as used in [the paper (Chin-Yew Lin)](https://aclanthology.org/W04-1013.pdf).
* [Answer Accuracy](./rageval/metrics/_answer_accuracy.py): also known as *Accuracy* as used in [the paper (Dan Hendrycks et al.)](https://arxiv.org/abs/2009.03300).
* [Answer LCS Ratio](./rageval/metrics/_answer_lcs_ratio.py): also know as *LCS(%)* as used in [the paper (Nashid et al.)](https://ieeexplore.ieee.org/abstract/document/10172590).
* [Answer Edit Distance](./rageval/metrics/_answer_edit_distance.py): also know as *Edit distance* as used in [the paper (Nashid et al.)](https://ieeexplore.ieee.org/abstract/document/10172590).

(2) **Answer Groundedness**: this category of metrics is to evaluate the groundedness (also known as factual consistency) by comparing the generated answer with the provided contexts. Here are some commonly used metrics:

* [Answer Citation Precision](./rageval/metrics/_answer_citation_precision.py): also known as *citation precision* in [the paper (Tianyu et al.)](https://arxiv.org/abs/2305.14627).
* [Answer Citation Recall](./rageval/metrics/_answer_citation_recall.py): also known as *citation recall* in [the paper (Tianyu et al.)](https://arxiv.org/abs/2305.14627).
* [Context Reject Rate](./rageval/metrics/_context_reject_rate.py): also known as *reject rate* in [the paper (Wenhao Yu et al.)](https://arxiv.org/abs/2311.09210).

### 2. [The rewrite task](./rageval/tasks/_rewrite.py)
The rewrite task is to reformulate user question into a set of queries, making them more friendly to the search module in RAG. 

### 3. [The search task](./rageval/tasks/_search.py)
The search task is to retrieve relevant documents from the knowledge base.

(1) **Context Adequacy**: this category of metrics is to evaluate the adequacy by comparing the retrieved documents with the groundtruth contexts. Here are some commonly used metrics:

(2) **Context Relevance**: this category of metrics is to evaluate the relevance by comparing the retrieved documents with the groundtruth answers. Here are some commonly used metrics:

* [Context Recall](./rageval/metrics/_context_recall.py): also known as *Context Recall* in [RAGAS framework](https://github.com/explodinggradients/ragas).

## Setup Evaluator LLMs

Some metrics evaluations rely on LLMs as evaluators. You can either directly call OpenAI's API or deploy an open-source model as a RESTful API in the OpenAI format for evaluation.

- OpenAI

```python
os.environ["OPENAI_API_KEY"] = "<your_openai_api_key>"
```


- Open source LLMs

Please use [vllm](https://github.com/vllm-project/vllm) to setup the API server for open source LLMs. For example, use the following command to deploy a Llama-3-8B model hosted on HuggingFace:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tensor-parallel-size 8 \
  --dtype auto \
  --api-key sk-123456789 \
  --gpu-memory-utilization 0.9 \
  --port 5000
```

## Benchmark Results

### 1. [ASQA benchmark](benchmarks/ASQA/README.md)

[ASQA dataset](https://huggingface.co/datasets/din0s/asqa) is a question-answering dataset that contains factoid questions and long-form answers. The benchmark evaluates the correctness of the answer in the dataset. All detailed results can be download from [this repo](https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa). Besides, these results can be reproduced based on [the script](./benchmarks/ASQA/run.sh) in this repo.


<table>
 <col width=166>
 <col width=125>
 <col width=125 span=4>
 <tr>
  <td rowspan=2 align="center"><b>Model</b></td>
  <td rowspan=2 align="center"><b>Retriever</b></td>
  <td colspan=4 align="center"><b>Metric</b></td>
 </tr>
 <tr>
  <td align="center"><a href="rageval\metrics\_answer_exact_match.py">String EM</a></td>
  <td align="center"><a href="rageval\metrics\_answer_rouge_correctness.py">Rouge L</a></td>
  <td align="center"><a href="rageval\metrics\_answer_disambig_f1.py">Disambig F1</a></td>
  <td align="center"><a href="benchmarks\ASQA\asqa_benchmark.py">D-R Score</a></td>
 </tr>
 <tr>
  <td>gpt-3.5-turbo-instruct</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/gpt_3.5_turbo_instruct">no-retrieval</a></td>
  <td align="center">33.8</td>
  <td align="center">30.2</td>
  <td align="center">30.7</td>
  <td align="center">30.5</td>
 </tr>
 <tr>
  <td>mistral-7b</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/mistral_7b">no-retrieval</a></td>
  <td align="center">20.6</td>
  <td align="center">31.1</td>
  <td align="center">26.6</td>
  <td align="center">28.7</td>
 </tr>
 <tr>
  <td>llama2-7b-chat</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/llama2_7b_chat">no-retrieval</a></td>
  <td align="center">21.7</td>
  <td align="center">30.7</td>
  <td align="center">28.0</td>
  <td align="center">29.3</td>
 </tr>
 <tr>
  <td>llama3-8b-base</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/llama3_8b_base">no-retrieval</a></td>
  <td align="center">25.7</td>
  <td align="center">31.0</td>
  <td align="center">28.4</td>
  <td align="center">29.7</td>
 </tr>
 <tr>
  <td>llama3-8b-instruct</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/llama3_8b_instruct">no-retrieval</a></td>
  <td align="center">27.1</td>
  <td align="center">30.9</td>
  <td align="center">29.4</td>
  <td align="center">30.1</td>
 </tr>
 <tr>
  <td>solar-10.7b-instruct</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/solar_10.7b_instruct">no-retrieval</a></td>
  <td align="center">23.0</td>
  <td align="center">24.9</td>
  <td align="center">28.1</td>
  <td align="center">26.5</td>
 </tr>
</table>

### 2. [ALCE Benchmark](benchmarks/ALCE)

[ALCE](https://github.com/princeton-nlp/ALCE) is a benchmark for Automatic LLMs' Citation Evaluation. ALCE contains three datasets: ASQA, QAMPARI, and ELI5. All detailed results can be download from [this repo](https://huggingface.co/datasets/golaxy/rag-bench/viewer/alce_eli5_bm25). Besides, these results can be reproduced based on [the script](./benchmarks/ALCE/ASQA/run.sh) in this repo.

For more evaluation results, please view the benchmark's README: [ALCE-ASQA](benchmarks/ALCE/ASQA/README.md) and [ALCE-ELI5](benchmarks/ALCE/ELI5/README.md).

<table>
 <col width=75>
 <col width=125>
 <col width=85>
 <col width=145>
 <col width=125 span=5>
 <tr>
  <td rowspan=2 align="center"><b>Dataset</b></td>
  <td rowspan=2 align="center"><b>Model</b></td>
  <td colspan=2 align="center"><b>Method</b></td>
  <td colspan=5 align="center"><b>Metric</b></td>
 </tr>
 <tr>
  <td align="center">retriever</td>
  <td align="center">prompt</td>
  <td align="center">MAUVE</td>
  <td align="center"><a href="rageval\metrics\_answer_exact_match.py">EM Recall</a></td>
  <td align="center"><a href="rageval\metrics\_answer_claim_recall.py">Claim Recall</a></td>
  <td align="center"><a href="rageval\metrics\_answer_citation_recall.py">Citation Recall</a></td>
  <td align="center"><a href="rageval\metrics\_answer_citation_precision.py">Citation Precision</a></td>
 </tr>
 <tr>
  <!-- <td rowspan=7><a href="benchmarks/ALCE/ASQA/README.md">ASQA</a></td>
  <td rowspan=7>llama2-7b-chat</td>
  <td rowspan=5>GTR</td>   -->
  <td rowspan=3 style="text-align:left;padding-left:10px"><a href="benchmarks/ALCE/ASQA/README.md">ASQA</a></td>
  <td rowspan=3>llama2-7b-chat</td>
  <td rowspan=1>GTR</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/alce_asqa_gtr">vanilla(5-psg)</a></td>
  <td align="center">-</td>
  <td align="center">33.3</td>
  <td align="center">-</td>
  <td align="center">55.9</td>
  <td align="center">80.0</td>
 </tr>
 <!-- <tr>
  <td>summary(5-psg)</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
 </tr>
  <tr>
  <td>summary(10-psg)</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
 </tr>
  <tr>
  <td>snippet(5-psg)</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
 </tr>
  <tr>
  <td>snippet(10-psg)</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
 </tr> -->
 <tr>
  <td>DPR</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/alce_asqa_dpr">vanilla(5-psg)</a></td>
  <td align="center">-</td>
  <td align="center">29.2</td>
  <td align="center">-</td>
  <td align="center">49.2</td>
  <td align="center">81.0</td>
 </tr>
 <tr>
  <td>Oracle</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/alce_asqa_oracle">vanilla(5-psg)</a></td>
  <td align="center">-</td>
  <td align="center">41.7</td>
  <td align="center">-</td>
  <td align="center">58.1</td>
  <td align="center">78.9</td>
 </tr>
 <tr>
  <!-- <td rowspan=6><a href="benchmarks/ALCE/ELI5/README.md">ELI5</a></td>
  <td rowspan=6>llama2-7b-chat</td>
  <td rowspan=5>BM25</td> -->
  <td rowspan=3><a href="benchmarks/ALCE/ELI5/README.md">ELI5</a></td>
  <td rowspan=3>llama2-7b-chat</td>
  <td rowspan=1>BM25</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/alce_eli5_bm25">vanilla(5-psg)</a></td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">11.5</td>
  <td align="center">26.6</td>
  <td align="center">74.5</td>
 </tr>
 <!-- <tr>
  <td>summary(5-psg)</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
 </tr>
  <tr>
  <td>summary(10-psg)</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
 </tr>
  <tr>
  <td>snippet(5-psg)</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
 </tr>
  <tr>
  <td>snippet(10-psg)</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
 </tr> -->
 <tr>
  <td>Oracle</td>
  <td><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/alce_eli5_oracle">vanilla(5-psg)</a></td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">17.8</td>
  <td align="center">34.0</td>
  <td align="center">75.6</td>
 </tr>
</table>


## Installation

```
git clone https://github.com/gomate-community/rageval.git
cd rageval
python setup.py install
```
## Usage

### 1. Metric

Take F1 as an example.
```
from datasets import Dataset
import rageval as rl

sample = {
    "answers": [
        "Democrat rick kriseman won the 2016 mayoral election, while re- publican former mayor rick baker did so in the 2017 mayoral election."
    ],
    "gt_answers": [
        [
            "Kriseman",
            "Rick Kriseman"
        ]
    ]
}
dataset = Dataset.from_dict(sample)
metric = rl.metrics.AnswerF1Correctness()
score, dataset = metric.compute(dataset)
```

### 2. Benchmark

Benchmarks can be run directly using scripts (Take ALCE-ELI5 as an example).
```
bash benchmarks/ALCE/ELI5/run.sh
```

## Contribution

Please make sure to read the [Contributing Guide](./CONTRIBUTING.md) before creating a pull request.

## About

This project is currently at its preliminary stage.
