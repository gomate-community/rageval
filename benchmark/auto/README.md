# AUTO BENCHMARK

Auto benchmark aims to generate testsets based on a provided corpus using the LLM. The corpus has two compoents: documents and few-shot cases.

## Usage

1. Prepare your corpus in a JSON file, following the format

```
corpus.json: [
      {"document": "Your document text here"}
]
few_shot_cases.json: [
      {"document": "Sample document", 
      "Query": "Sample question"}
]
```

2. Place the corpus JSON file(s) in `corpus` directory.
3. Run `run.sh` to start dataset generation. The result will saved in `output` directory.

Arguments:
`--corpus_dir`: Directory containing the corpus and few-shot case JSON files.
`--output_dir`: Directory where the generated dataset JSON will be saved.
`--model`: The OpenAI GPT model to use, e.g., gpt-3.5-turbo-16k.
`--api_key`: Your OpenAI API key.

## Citations

In this case, `documents` refers to a list of news articles, and `few-shot cases` are derived from a random split of the Multi-RC dataset. And we refer to the prompt from ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.

``` bibtex
@misc{saadfalcon2023ares,
      title={ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems}, 
      author={Jon Saad-Falcon and Omar Khattab and Christopher Potts and Matei Zaharia},
      year={2023},
      eprint={2311.09476},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{MultiRC2018,
    author = {Daniel Khashabi and Snigdha Chaturvedi and Michael Roth and Shyam Upadhyay and Dan Roth},
    title = {Looking Beyond the Surface:A Challenge Set for Reading Comprehension over Multiple Sentences},
    booktitle = {Proceedings of North American Chapter of the Association for Computational Linguistics (NAACL)},
    year = {2018}
}
```
