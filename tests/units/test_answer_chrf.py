import pytest
from datasets import Dataset

from rageval.metrics import AnswerCHRFCorrectness


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "The relationship between cats and dogs is not exactly friendly.",
            "a good bookshop is just a genteel black hole that knows how to read."
        ],
        "gt_answers": [
            ["The relationship between dogs and cats is not exactly friendly.", ],
            ["A good bookshop is just a genteel Black Hole that knows how to read."]
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.mark.slow
def test_case_on_answer_ter(testset):
    metric = AnswerCHRFCorrectness()
    assert metric.name == "answer_chrf"
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset['answers'], testset['gt_answers'], 1)
