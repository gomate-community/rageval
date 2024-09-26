import pytest
from datasets import Dataset

from rageval.metrics import AnswerF1Correctness


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "Ali Dael has the highest goals in men's world international football with 109 goals. Josef Bican has the \
            highest goals all-time in men's football and Christine Sinclair has the highest goals in women's world \
            international football.",
            "A supercentenarian is someone who has reached the age of 110. Sarah Knauss, whose age is undisputed, was \
            the oldest person ever from the United States and the second-oldest fully documented person ever. Jeanne \
            Calment was a French supercentenarian and the oldest human whose age is well-documented, with a lifespan \
            of 122 years and 164 days, and was the oldest person in the world as of 1997."
        ],
        "gt_answers": [
            ["Daei", "Ali Daei"],
            ["Jeanne Calment"]
        ],
        "answers_zh": [
            "魏晋",
            "北齐只设于清都郡。",
        ],
        "gt_answers_zh": [
            ["魏晋", "魏晋时期"],
            ["北齐只设于清都郡。", "清都郡"]
        ],
        "answers_num":[[1,2,3], [4,5,6]],
        "gt_answers_num":[[2,3,4,5,6], [1,2,3,4,5]]
    }
    return test_case

@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds

@pytest.mark.slow
def test_case_on_answer_f1(testset):
    metric = AnswerF1Correctness(normalize=True, language='en')
    assert metric.name == "answer_f1"
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset['answers'], testset['gt_answers'])
    assert 0 <= score <= 1

    metric = AnswerF1Correctness(normalize=True, language='zh')
    assert metric.name == "answer_f1"
    assert metric.mtype == 'AnswerCorrectness'
    score_zh, results_zh = metric.compute(testset['answers_zh'], testset['gt_answers_zh'])
    assert 0 <= score_zh <= 1

    metric = AnswerF1Correctness(normalize=False)
    assert metric.name == "answer_f1"
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset['answers_num'], testset['gt_answers_num'])
    assert 0 <= score <= 1
