# Rageval

Evaluation tools for Retrieval-augmented Generation (RAG) methods.

[![python](https://img.shields.io/badge/Python-3.8.18-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
![workflow status](https://github.com/gomate-community/rageval/actions/workflows/makefile.yml/badge.svg)
[![codecov](https://codecov.io/gh/gomate-community/rageval/graph/badge.svg?token=AH4DNR46HL)](https://codecov.io/gh/gomate-community/rageval)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

Rageval is a tool that helps you evaluate RAG system. The evaluation consists of six sub-tasks, including query rewriting, document ranking, information compression, evidence verify, answer generating, and result validating.

## Definition of tasks and metrics
### Generator
After obtaining relevant documents pieces, the generator is tasked with answer the question by utilizing the original user query and the retrieved contexts. We assess a generator module from two distinct perspectives.
(1) Answer Correctness
In this task, we compare output answer with groundtruth answer using following metrics. 
* answer claim recall ("answer_claim_recall")
* answer exact match ("answer_exact_match")
* context reject rate ("context_reject_rate")
(2) Answer Groundednedd
* answer_citation_precision ("answer_citation_precision")
* answer_citation_recall ("answer_citation_recall")

### Rewriter
xxx




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
