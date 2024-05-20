import pytest
from datasets import Dataset

from rageval.metrics import AnswerDisambigF1Correctness


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "Ali Daei holds the record for the most international goals in world football according to FIFA. Josef Bican holds the record for the most total goals in world football according to UEFA.",
        ],
        "gt_answers": [
            ["Ali Dael has the highest goals in men's world international football with 109 goals. Josef Bican has the highest goals all-time in men's football and Christine Sinclair has the highest goals in women's world international football.",
            "The players with the highest all-time goals and highest men's and women's international football goals differ. The player with the highest all-time men's football goals is Josef Bican, who in 2020 was recognized by FIFA, the international governing body of football, as the record scorer with an estimated 805 goals. Christine Sinclair has the highest goals in women's international football with 187 and is the all-time leader for international goals scored for men or women. Cristiano Ronaldo and Ali Daei are currently tied for leading goalscorer in the history of men's international football with 109."],
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.mark.slow
def test_case_on_answer_disambig_f1(testset):
    metric = AnswerDisambigF1Correctness()
    assert metric.name == "answer_disambig_f1"
    assert metric.homepage == ""
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(1, testset['answers'], testset['gt_answers'])
    assert 0 <= score <= 1
    assert isinstance(results, Dataset)
