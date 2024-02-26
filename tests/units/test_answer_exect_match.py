import pytest
from datasets import Dataset

from rageval.metrics import AnswerExactMatch


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "Ali Dael has the highest goals in men's world international football with 109 goals. Josef Bican has the highest goals all-time in men's football and Christine Sinclair has the highest goals in women's world international football.",
            "A supercentenarian is someone who has reached the age of 110. Sarah Knauss, whose age is undisputed, was the oldest person ever from the United States and the second-oldest fully documented person ever. Jeanne Calment was a French supercentenarian and the oldest human whose age is well-documented, with a lifespan of 122 years and 164 days, and was the oldest person in the world as of 1997. In 1985, the oldest living person was Mathew Beard and in 1986 it was Augusta Holtz, who lived 115 years and 79 days, from 1871 to 1986."
        ],
        "gt_answers": [
            [
                ["Daei", "Ali Daei"],
                ["Bican", "Josef Bican"],
                ["Sinclair","Christine Sinclair"]
            ],
            [
                ["Jeanne Calment"],
                ["Sarah Knauss"],
                ["Augusta-Holtz"],
            ]
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


def test_case_on_answer_exact_match(testset):
    metric = AnswerExactMatch()
    assert metric.name == "answer_exact_match"
    assert metric.homepage == ""
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset, 1)
    assert 0 <= score <= 1
    assert isinstance(results, Dataset)
