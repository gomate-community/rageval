import logging
from abc import ABC

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class Tokenizer(ABC):
    """This is the hugging face tokenizer model."""

    def __init__(self, model: str = "Qwen/Qwen2-0.5B-Instruct") -> None:
        """Init the Model."""
        #self._model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
