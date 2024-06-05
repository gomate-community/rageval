import pytest
from datasets import Dataset

from rageval.metrics import AnswerAccuracy


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "A",
            "B",
            "C"
        ],
        "gt_answers": [
            "A",
            "C",
            "C"
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.mark.slow
def test_case_on_answer_accuracy(testset):
    metric = AnswerAccuracy()
    assert metric.name == "answer_accuracy"
    assert metric.mtype == 'AnswerCorrectness'
    assert repr(metric) == "answer_accuracy"
    score, results = metric.compute(testset["answers"], testset["gt_answers"], 1)
    assert score == 2 / 3
    assert results[0] is True
