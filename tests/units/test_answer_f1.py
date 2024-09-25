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
        "答案": [
            "新品研发效率提升56.6%、不合格率下降47.3%、机时产量提升15%，精确的数据展现，使得钢铁行业在“智慧大脑”的调度下如虎添翼。",
            "快手大模型团队负责人表示，技术革新正极大地降低视频内容制作的门槛，让更多有创意的人不再受限于设备和成本，凭借创造力和想象力就可以进行视频生产。"
        ],
        "参考答案": [
            ["智慧大脑", "钢铁行业"],
            ["快手大模型技术革新"]
        ]
    }
    return test_case

@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds

@pytest.mark.slow
def test_case_on_answer_f1(testset):
    metric = AnswerF1Correctness(normalize=True)
    assert metric.name == "answer_f1"
    assert metric.mtype == 'AnswerCorrectness'
    score, results = metric.compute(testset['answers'], testset['gt_answers'], 1)
    score_zh, results_zh = metric.compute(testset['答案'], testset['参考答案'], 1,  ['Chinese'] * len(testset['答案']))

    assert 0 <= score <= 1
<<<<<<< HEAD
    score = metric._compute_one(testset['answers'][0], testset['gt_answers'][0])
    assert 0 <= score <= 1
=======
    assert 0 <= score_zh <= 1

>>>>>>> 1120c938e9cecf4b86cb5e93821937ea4a6e26e3
