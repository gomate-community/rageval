# ASQA BENCHMARK

## Description

This benchmark is designed to evaluate the performance of the [ASQA dataset](https://huggingface.co/datasets/din0s/asqa). For `generate.py`, we followed [FLARE](https://github.com/jzbjyb/FLARE), using `gpt-3.5-turbo-instruct` with no retrieval settings.

## Dataset

TODO

## Usage

1. Prepare your model output results in the `output` directory. You can just attatch your `answers` to the end of each example in the origin ASQA dataset as what we did in the `dataset.json` file.
2. Run `run.sh` to start dataset generation. The result will be saved in `output` directory, named `results.json`. And the `result_datasets.json` file is the json dump of the detailed result.

### Arguments:

`--max_num_examples`: The maximum number of examples used in generate answers.

`--max_new_tokens`: The maximum number of tokens that can be generated for answering questions.

`--output_dir`: Directory where the generated answers and the final results will be saved.

`--model`: The OpenAI GPT model to use, e.g., gpt-3.5-turbo-instruct.

`--api_key`: Your OpenAI API key.

## Metrics

## Performance

| Model | Rouge-L | Exact Match | Disambig F1 | D-R Score|
|:---:|:---:|:---:|:---:|:---:|
| gpt-3.5-turbo-instruct | 29.8 | 34.0 | 30.6 | 30.2 |
| text-davinci-003 (FLARE) | 33.3 | 33.8 | 24.2 | 28.4 |

## Citations

``` bibtex
@article{stelmakh2022asqa,
  title={ASQA: Factoid questions meet long-form answers},
  author={Stelmakh, Ivan and Luan, Yi and Dhingra, Bhuwan and Chang, Ming-Wei},
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
```
