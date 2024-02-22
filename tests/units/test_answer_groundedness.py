"""Test the AnswerGroundedness Metric."""

# -*- coding: utf-8 -*-

import os
import pytest
import pandas as pd
from datasets import Dataset

from rageval.models import NLIModel
from rageval.metrics import AnswerGroundedness

@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": ["In 1980, the oldest justice on the United States Supreme Court was Justice William O. Douglas."],
        "contexts": [["August 3, 1994 \u2013 June 30, 2022 (27 years, 10 months, 27 days) photo source: Wikimedia Commons After the passing of Ruth Bader Ginsberg in 2020, Stephen Breyer was the oldest sitting member of the Supreme Court until his retirement in 2022. Stepping down at the age of 83, Breyer is now one of the oldest Supreme Court justices ever. Breyer was nominated by Bill Clinton and served on the Court for more than 27 years. During his tenure, Breyer fell in line with the liberal wing of the court. Before he was appointed to the Supreme Court, Breyer served as a judge on the U.S. Court of Appeals for the First Circuit; he was the Chief Judge for the last four years of his appointment."]]
    }
    return test_case

@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds

def test_case_on_answer_groundedness_metric(testset):
    metric = AnswerGroundedness()
    model = NLIModel('text-classification', 'hf-internal-testing/tiny-random-RobertaPreLayerNormForSequenceClassification')
    metric.init_model(model)
    score, results = metric.score(testset, 1)
    assert score == 0 or score == 1
    assert isinstance(results, Dataset)
