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
def test_case_on_answer_lcs_ratio(testset):
    metric = AnswerAccuracy()
    assert metric.name == "answer_accuracy"
    assert metric.homepage == ""
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset, 1)
    assert score == 2 / 3
    assert isinstance(results, Dataset)
