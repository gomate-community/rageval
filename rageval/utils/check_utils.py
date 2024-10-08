import logging
import re
from typing import List

import nltk
from nltk.downloader import Downloader

from rageval.models import OpenAILLM
from .prompt import DOC_TO_SENTENCES_PROMPT

logger = logging.getLogger(__name__)
if not Downloader().is_installed('punkt_tab'):
    nltk.download('punkt_tab')


def text_to_sents(text: str, model_name="nltk") -> List[str]:
    """Convert the text into a set of sentences."""
    sentences = []
    if model_name == "nltk":
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 3]

    elif model_name == "gpt-3.5-turbo":
        model = OpenAILLM("gpt-3.5-turbo", "OPENAI_API_KEY")  # pragma: no cover
        prompt = DOC_TO_SENTENCES_PROMPT  # pragma: no cover
        input_str = prompt.format(doc=text).strip()  # pragma: no cover
        r = model.generate([input_str])  # pragma: no cover
        sentences = eval(r)  # pragma: no cover
    else:
        logger.info("The parameter `model_name` should be in [`nltk`, `gpt-3.5-turbo`]. ")  # pragma: no cover
    assert isinstance(sentences, list)

    return sentences


def remove_citations(text: str) -> str:
    """Remove the citation in the text."""
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", text)).replace(" |", "").replace("]", "")
