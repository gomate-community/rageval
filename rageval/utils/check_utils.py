import logging
import re
from typing import List

import nltk

from rageval.models import OpenAILLM
from .prompt import DOC_TO_SENTENCES_PROMPT

logger = logging.getLogger(__name__)
nltk.download('punkt')

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
        logger.info("The parameter `model_name` should be in [`nltk`, `gpt-3.5-turbo-16k`]. ")
    assert isinstance(sentences, list)

    return sentences


def remove_citations(text: str) -> str:
    """Remove the citation in the text."""
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", text)).replace(" |", "").replace("]", "")
