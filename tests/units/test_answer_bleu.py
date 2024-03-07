import pytest
from datasets import Dataset

from rageval.metrics import AnswerBleuScore


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "Hello, world!",
            "I am a metric named bleu."
        ],
        "gt_answers": [
            [
                "Hello, my world!",
            ],
            [
                "I am a metric named bleu.",
                "I am bleu metric.",
            ]
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.mark.slow
def test_case_on_answer_bleu(testset):
    metric = AnswerBleuScore()
    assert metric.name == "answer_bleu"
    assert metric.homepage == ""
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset, 1)
    assert 0.0 <= score <= 1.0
    assert isinstance(results, Dataset)
