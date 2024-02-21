# -*- coding: utf-8 -*-

import os
import logging
import pytest
from abc import ABC
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)


class NLIModel(ABC):
    """This is the Roberta-based NLI model."""

    def __init__(self, task: str = "sentiment-analysis", model: str = "roberta-large-mnli") -> None:
        """Init the Roberta Model."""
        self._model_name = model
        self._model = pipeline(task=task, model=model)

        self._labelmap = {
            "NEUTRAL": 3,
            "CONTRADICTION": 2,
            "ENTAILMENT": 1
        }
        self._nli2stance = {
            "NEUTRAL": "irrelevant",
            "CONTRADICTION": "refute",
            "ENTAILMENT": "support"
        }
        self._stancemap = {
            'irrelevant': 3,
            'refute': 2,
            'partially-support': 1,
            'completely-support': 1
        }

    @property
    def model(self):
        """Construct the OpenAI LLM model."""
        return self._model

    @pytest.mark.api
    def infer_prob(self, premise, hypothesis):
        """Predit one sample with NLI model."""
        try:
            input = "<s>{}</s></s>{}</s></s>".format(premise, hypothesis)
            pred = self._model(input)
            # print(pred)
        except Exception as e:
            # token length > 514
            L = len(premise)
            premise = premise[:int(L / 2)]
            input = "<s>{}</s></s>{}</s></s>".format(premise, hypothesis)
            pred = self._model(input)
            logger.info(f"An exception occured during nli inference: {e}")
        return pred

    @pytest.mark.api
    def infer(self, premise, hypothesis):
        """Predct one sample with NLI model."""
        pred = self.infer_prob(premise, hypothesis)
        # [{'label': 'CONTRADICTION', 'score': 0.9992701411247253}]
        if 'mnli' in self._model_name:
            return self._nli2stance[pred[0]['label']]
        else:
            nli2stance = {
                "LABEL_0": "irrelevant",
                "LABEL_1": "support"
            }
            return nli2stance[pred[0]['label']]
