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

## Benchmark

<style id="readme_22070_Styles">
.xl6522070
	{padding-top:1px;
	padding-right:1px;
	padding-left:1px;
    padding-bottom:1px;
	mso-ignore:padding;
	color:black;
	font-size:11.0pt;
	font-weight:400;
	font-style:normal;
	text-decoration:none;
	mso-font-charset:0;
	mso-number-format:General;
	text-align:center;
	vertical-align:middle;
	mso-background-source:auto;
	mso-pattern:auto;
	white-space:nowrap;}
.xl6622070
	{padding-top:1px;
	padding-right:1px;
	padding-left:1px;
    padding-bottom:1px;
	mso-ignore:padding;
	color:black;
	font-size:11.0pt;
	font-weight:400;
	font-style:normal;
	text-decoration:none;
	mso-font-charset:0;
	mso-number-format:"0\.0";
	text-align:center;
	vertical-align:middle;
	mso-background-source:auto;
	mso-pattern:auto;
	white-space:nowrap;}
</style>

ASQA benchmark:

<table border=0 cellpadding=0 cellspacing=0 width=100% style='border-collapse:
 collapse;table-layout:fixed;width:763pt'>
 <col width=166>
 <col width=125>
 <col width=125 span=4>
 <tr height=18 style='height:13.8pt'>
  <td rowspan=2 height=36 class=xl6522070 style='height:27.6pt;font-weight:600'>Model</td>
  <td rowspan=2 class=xl6522070 style='font-weight:600'>Method</td>
  <td colspan=4 class=xl6522070 style='font-weight:600'>Metric</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt'>String EM</td>
  <td class=xl6522070>Rouge L</td>
  <td class=xl6522070>Disambig F1</td>
  <td class=xl6522070>D-R Score</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt;text-align:left;padding-left:10px;'><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/gpt_3.5_turbo_instruct">gpt-3.5-turbo-instruct</a></td>
  <td class=xl6522070>no-retrieval</td>
  <td class=xl6622070>33.8</td>
  <td class=xl6622070>30.2</td>
  <td class=xl6622070>30.7</td>
  <td class=xl6622070>30.5</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt;text-align:left;padding-left:10px;'><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/mistral_7b">mistral-7b</a></td>
  <td class=xl6522070>no-retrieval</td>
  <td class=xl6622070>20.6</td>
  <td class=xl6622070>31.1</td>
  <td class=xl6622070>26.6</td>
  <td class=xl6622070>28.7</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt;text-align:left;padding-left:10px;'><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/llama2_7b_chat">llama2-7b-chat</a></td>
  <td class=xl6522070>no-retrieval</td>
  <td class=xl6622070>21.7</td>
  <td class=xl6622070>30.7</td>
  <td class=xl6622070>28.0</td>
  <td class=xl6622070>29.3</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt;text-align:left;padding-left:10px;'><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/solar_10.7b_instruct">solar-10.7b-instruct</a></td>
  <td class=xl6522070>no-retrieval</td>
  <td class=xl6622070>23.0</td>
  <td class=xl6622070>24.9</td>
  <td class=xl6622070>28.1</td>
  <td class=xl6622070>26.5</td>
</table>

ELI5(ALCE) Benchmark:
<table border=0 cellpadding=0 cellspacing=0 width=100% style='border-collapse:
 collapse;table-layout:fixed;width:763pt'>
 <col width=166>
 <col width=125>
 <col width=125 span=4>
 <tr height=18 style='height:13.8pt'>
  <td rowspan=2 height=36 class=xl6522070 style='height:27.6pt;font-weight:600'>Model</td>
  <td rowspan=2 class=xl6522070 style='font-weight:600'>Method</td>
  <td colspan=4 class=xl6522070 style='font-weight:600'>Metric</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt'>String EM</td>
  <td class=xl6522070>Rouge L</td>
  <td class=xl6522070>Disambig F1</td>
  <td class=xl6522070>D-R Score</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt;text-align:left;padding-left:10px;'><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/gpt_3.5_turbo_instruct">gpt-3.5-turbo-instruct</a></td>
  <td class=xl6522070>no-retrieval</td>
  <td class=xl6622070>33.8</td>
  <td class=xl6622070>30.2</td>
  <td class=xl6622070>30.7</td>
  <td class=xl6622070>30.5</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt;text-align:left;padding-left:10px;'><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/mistral_7b">mistral-7b</a></td>
  <td class=xl6522070>no-retrieval</td>
  <td class=xl6622070>20.6</td>
  <td class=xl6622070>31.1</td>
  <td class=xl6622070>26.6</td>
  <td class=xl6622070>28.7</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt;text-align:left;padding-left:10px;'><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/llama2_7b_chat">llama2-7b-chat</a></td>
  <td class=xl6522070>no-retrieval</td>
  <td class=xl6622070>21.7</td>
  <td class=xl6622070>30.7</td>
  <td class=xl6622070>28.0</td>
  <td class=xl6622070>29.3</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl6522070 style='height:13.8pt;text-align:left;padding-left:10px;'><a href="https://huggingface.co/datasets/golaxy/rag-bench/viewer/asqa/solar_10.7b_instruct">solar-10.7b-instruct</a></td>
  <td class=xl6522070>no-retrieval</td>
  <td class=xl6622070>23.0</td>
  <td class=xl6622070>24.9</td>
  <td class=xl6622070>28.1</td>
  <td class=xl6622070>26.5</td>
</table>




## Installation

```
git clone https://github.com/gomate-community/rageval.git
cd rageval
python setup.py install
```
## Usage

```
import rageval as rl

test_set = rl.datasets.load_data('ALCE', task='')
metric = rl.metrics.ContextRecall()
model = rl.models.OpenAILLM()
metric.init_model(model)

results = metric._score_batch(teset_set)

```

## Contribution

Please make sure to read the [Contributing Guide](./CONTRIBUTING.md) before creating a pull request.

## About

This project is currently at its preliminary stage.
