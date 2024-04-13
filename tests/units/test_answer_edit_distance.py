import pytest
from datasets import Dataset

from rageval.metrics import AnswerEditDistance


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "Language models trained on massive code corpora can generalize to tasks without the need "
            "for task-specific fine tuning."
        ],
        "gt_answers": [
            "Large language models trained on massive code corpora can generalize to new tasks without the need "
            "for task-specific fine-tuning."
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.mark.slow
def test_case_on_answer_edit_distance(testset):
    metric = AnswerEditDistance()
    assert metric.name == "answer_edit_distance"
    assert metric.homepage == ""
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset, 1)
    assert score == 5 / 18
    assert isinstance(results, Dataset)
