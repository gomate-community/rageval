import pytest
from datasets import Dataset

from rageval.metrics import AnswerDistinct12


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "hypothesis": [
            "The relationship between cats and dogs is not exactly friendly.",
            "A good bookshop is just a genteel black hole that knows how to read.",
            "The quick brown fox jumps over the lazy dog."
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.mark.slow
def test_case_on_distinct(testset):
    metric = AnswerDistinct12()
    assert metric.name == "distinct"
    assert metric.mtype == 'Diversity'
    assert repr(metric) == "distinct"
    scores = metric.compute(testset["hypothesis"])
    assert "distinct_1" in scores
    assert "distinct_2" in scores
    assert isinstance(scores["distinct_1"], float)
    assert isinstance(scores["distinct_2"], float)
