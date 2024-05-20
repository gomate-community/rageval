import pytest
from datasets import Dataset
from langchain.llms.fake import FakeListLLM

from rageval.metrics import ContextRejectRate


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "questions": [
            "Why did Bushnell set himself on fire?",
            "Did Bushnell have a wife?"
        ],
        "contexts": [
            [
                [
                    "An active-duty member of the U.S. Air Force has died after he set himself ablaze outside the "
                    "Israeli Embassy in Washington, D.C., while declaring that he “will no longer be complicit in "
                    "genocide.”"
                ],
                [
                    "The 25-year-old airman, Aaron Bushnell, of San Antonio, Texas, died from his injuries, the "
                    "Metropolitan Police Department said Monday."
                ],
                [
                    "Bushnell had walked up to the embassy shortly before 1 p.m. Sunday and began livestreaming on "
                    "the video streaming platform Twitch, a person familiar with the matter told The Associated "
                    "Press. Law enforcement officials believe he set his phone down and then doused himself in "
                    "accelerant and ignited the flames. At one point, he said he “will no longer be complicit in "
                    "genocide,” the person said. The video was later removed from the platform, but law enforcement "
                    "officials have obtained and reviewed a copy."
                ]
            ],
            [
                [
                    "An active-duty member of the U.S. Air Force has died after he set himself ablaze outside the "
                    "Israeli Embassy in Washington, D.C., while declaring that he “will no longer be complicit in "
                    "genocide.”"
                ],
                [
                    "The 25-year-old airman, Aaron Bushnell, of San Antonio, Texas, died from his injuries, the "
                    "Metropolitan Police Department said Monday."
                ],
                [
                    "Bushnell had walked up to the embassy shortly before 1 p.m. Sunday and began livestreaming on "
                    "the video streaming platform Twitch, a person familiar with the matter told The Associated "
                    "Press. Law enforcement officials believe he set his phone down and then doused himself in "
                    "accelerant and ignited the flames. At one point, he said he “will no longer be complicit in "
                    "genocide,” the person said. The video was later removed from the platform, but law enforcement "
                    "officials have obtained and reviewed a copy."
                ]
            ]
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.mark.slow
def test_case_on_context_reject_rate(testset):
    model = FakeListLLM(
        responses=[
            "Answer: wrong response format",
            "Answer: sorry, cannot answer the question"
        ]
    )
    metric = ContextRejectRate(model)
    assert metric.name == "context_reject_rate"
    assert metric.homepage == ""
    assert metric.mtype == 'AnswerGroundedness'
    score, results = metric.compute(testset['answers'], testset['gt_answers'])
    assert score == 0.5
    assert isinstance(results, Dataset)
