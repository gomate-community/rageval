"""Test the AnswerClaimRecall Metric."""

import pytest
from datasets import Dataset

from rageval.models import NLIModel
from rageval.metrics import AnswerNLICorrectness


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "Yes. Did you watch The Social Network? They went a while before introducing ads, so they could make "
            "money, as they needed to establish their brand and amass users. Once you have dedicated users, "
            "introducing ads won't deter most, but if you are still new, having ads will deter a lot. The same goes "
            "for Uber, it's not that they aren't making money, it's that they are reinvesting a ton of it to make "
            "their service better."
        ],
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
def sample_with_decompose():
    test_case = {
        "answers": [
            "Yes. Did you watch The Social Network? They went a while before introducing ads, so they could make \
             money, as they needed to establish their brand and amass users. Once you have dedicated users, \
             introducing ads won't deter most, but if you are still new, having ads will deter a lot. The same goes \
             for Uber, it's not that they aren't making money, it's that they are reinvesting a ton of it to make \
             their service better."
        ],
        "gt_answers": [
            "Firms like Snapchat and Uber need to establish their brand and amass users before introducing ads. \
            Introducing ads too early can deter potential users. Uber is reinvesting a lot of money to make their \
            service better."
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.fixture(scope='module')
def testset_with_decompose(sample_with_decompose):
    ds = Dataset.from_dict(sample_with_decompose)
    return ds


@pytest.mark.slow
def test_case_on_answer_claim_recall_metric(testset):
    nli_model = NLIModel(
        'text2text-generation',
        'hf-internal-testing/tiny-random-T5ForConditionalGeneration'
    )
    metric = AnswerNLICorrectness(nli_model=nli_model)
    assert metric.name == "answer_claim_recall"
    assert metric.homepage == ""
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(1, testset['answers'], testset['gt_answers'])
    assert score == 0 or score == 1
    assert isinstance(results, Dataset)


@pytest.mark.slow
def test_case_on_answer_claim_recall_metric_with_decompose(testset_with_decompose):
    nli_model = NLIModel(
        'text2text-generation',
        'hf-internal-testing/tiny-random-T5ForConditionalGeneration'
    )
    metric = AnswerNLICorrectness(nli_model=nli_model, decompose_model="nltk")
    assert metric.name == "answer_claim_recall"
    assert metric.homepage == ""
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(1, testset_with_decompose['answers'], testset_with_decompose['gt_answers'])
    assert score == 0 or score == 1
    assert isinstance(results, Dataset)
