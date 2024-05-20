import pytest
from datasets import Dataset

from rageval.metrics import AnswerTERCorrectness


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "does this sentence match??",
            "what about this sentence?",
            "What did the TER metric user say to the developer?"
        ],
        "gt_answers": [
            ["does this sentence match", "does this sentence match!?!"],
            ["wHaT aBoUt ThIs SeNtEnCe?", "wHaT aBoUt ThIs SeNtEnCe?"],
            ["Your jokes are...", "...TERrible"]
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.mark.slow
def test_case_on_answer_ter(testset):
    metric = AnswerTERCorrectness()
    assert metric.name == "answer_ter"
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset['answers'], testset['gt_answers'])
    assert isinstance(results, Dataset)
