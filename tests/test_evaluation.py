"""Test the evaluation function."""

import sys
sys.path.insert(0, '../src')
import pytest

#import rageval

@pytest.mark.cron
def test_evaluation():
    '''
    This is test unit for testing the load_dataset function.
    '''

    # 1) init test task: task type, metrics

    # 2) load dataset, and extract testset
    #train_data = rageval.datasets.load_data('', task='')
    #assert len(train_data) == 300

    # 3) run evaluation
    # result = evaluate(testset, info)
