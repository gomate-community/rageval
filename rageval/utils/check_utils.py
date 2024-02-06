# -*- coding: utf-8 -*-

import json
import os
import nltk
import logging
from typing import List

from rageval.models import OpenAILLM
from .prompt import DOC_TO_SENTENCES_PROMPT

logger = logging.getLogger(__name__)


def text_to_sents(text: str, model_name="nltk") -> List[str]:
    """Convert the text into a set of sentences."""
    sentences = []
    if model_name == "nltk":
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 3]

    elif model_name == "gpt-3.5-turbo":
        model = OpenAILLM("gpt-3.5-turbo-16k", "OPENAI_API_KEY")
        prompt = DOC_TO_SENTENCES_PROMPT
        input_str = prompt.format(doc=text).strip()
        r = model.generate([input_str])
        sentences = eval(r)
    else:
        logger.info("The paramerter `model_name` should be in [`nltk`, `gpt-3.5-turbo-16k`]. ")
    assert isinstance(sentences, list)

    return sentences
