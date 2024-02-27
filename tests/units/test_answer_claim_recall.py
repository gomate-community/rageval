"""Test the AnswerClaimRecall Metric."""

# -*- coding: utf-8 -*-

import pytest
from datasets import Dataset

from rageval.models import NLIModel
from rageval.metrics import AnswerClaimRecall

@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": ["Yes. Did you watch The Social Network? They went a while before introducing ads, so they could make money, as they needed to establish their brand and amass users. Once you have dedicated users, introducing ads won't deter most, but if you are still new, having ads will deter a lot. The same goes for Uber, it's not that they aren't making money, it's that they are reinvesting a ton of it to make their service better."],
        "gt_answers": [
            [
                "Firms like Snapchat and Uber need to establish their brand and amass users before introducing ads.",
                "Introducing ads too early can deter potential users.",
                "Uber is reinvesting a lot of money to make their service better."
            ]
        ]
    }
    return test_case

@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds

def test_case_on_answer_groundedness_metric(testset):
    metric = AnswerClaimRecall()
    model = NLIModel('text-classification', 'hf-internal-testing/tiny-random-RobertaPreLayerNormForSequenceClassification')
    metric.init_model(model)
    assert metric.name == "answer_claim_recall"
    assert metric.homepage == ""
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset, 1)
    assert score == 0 or score == 1
    assert isinstance(results, Dataset)
