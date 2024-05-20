import pytest
from datasets import Dataset

from rageval.metrics import AnswerBleuScore


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "It is a guide to action which ensures that the military always obeys the commands of the party.",
            "It is to insure the troops forever hearing the activity guidebook that party direct."
        ],
        "gt_answers": [
            [
                "It is a guide to action that ensures that the military will forever heed Party commands.",
                "It is the guiding principle which guarantees the military forces always being under the command of the Party.",
                "It is the practical guide for the army always to heed the directions of the party."
            ],
            [
                "It is a guide to action that ensures that the military will forever heed Party commands.",
                "It is the guiding principle which guarantees the military forces always being under the command of the Party.",
                "It is the practical guide for the army always to heed the directions of the party."
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
    score, results = metric.compute(1, testset['answers'], testset['gt_answers'])
    assert score == 0.3172992057845065
    assert isinstance(results, Dataset)
