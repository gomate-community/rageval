import dataclasses
from typing import List, Tuple

import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import logging
from rageval.metrics import Metric, add_attribute

_DESCRIPTION = """\
Perplexity (PPL) is one of the most common metrics for evaluating language models.
It is defined as the exponentiated average negative log-likelihood of a sequence, calculated with exponent base `e`.

For more information, see https://huggingface.co/docs/transformers/perplexity
"""

_KWARGS_DESCRIPTION = """\
Args:
    model_id (str): model used for calculating Perplexity
            NOTE: Perplexity can only be calculated for causal language models.
                    This includes models such as gpt2, causal variations of bert,
                    causal versions of t5, and more (the full list can be found
                    in the AutoModelForCausalLM documentation here:
                    https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForCausalLM )

    predictions (list of str): input text, each separate text snippet
        is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
    device (str): device to run on, defaults to 'cuda' when available

Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "texts": [
    ...         "The relationship between cats and dogs is not exactly friendly.",
    ...         "A good bookshop is just a genteel black hole that knows how to read."
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.Perplexity()
    >>> metric.mtype
    'LanguageModelQuality'
    >>> score = metric.compute(model_id='gpt2', predictions=dataset['texts'], batch_size=1)
    >>> score['mean_perplexity']
    647.0
"""

_CITATION = """\
@misc{HuggingFace2022perplexity,
      title={Perplexity Metric for Language Models},
      author={HuggingFace Datasets Authors},
      year={2022},
}
"""


@dataclasses.dataclass
@add_attribute('mtype', 'LanguageModelQuality')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Perplexity(Metric):
    """Perplexity metric for language models."""

    name = "perplexity"

    ALIAS = ['perplexity']

    model_id: str
    batch_size: int = 16
    add_start_token: bool = True
    device: str = None
    max_length: int = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"]
        )

    def _compute(
        self,
        predictions: List[str],
        pipeline,
    ) -> Tuple[float, List[float]]:
        """Compute perplexity scores for the input texts."""
        if self.device not in ["gpu", "cpu", "cuda"]:
            raise ValueError("device should be either gpu or cpu.")
        model = pipeline.model
        tokenizer = pipeline.tokenizer

        if tokenizer.pad_token is None and self.batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            if len(existing_special_tokens) == 0:
                raise ValueError("Model must have at least one special token to use for padding when batch_size > 1.")
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        max_tokenized_len = self.max_length - 1 if self.add_start_token and self.max_length else self.max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        if self.add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(torch.ge(attn_masks.sum(1), 2)), "Each input text must be at least two tokens long if add_start_token=False."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), self.batch_size)):
            end_index = min(start_index + self.batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if self.add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat([torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1)

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1) / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

    def _compute_batch(
        self,
        pred_answers: List[str],
        ref_answers: List[List[str]]
    ) -> List[float]:
        pass
