import sys
sys.path.insert(0, '../../src')

import pytest
import numpy as np

#from rageval import utils

@pytest.fixture
def raw_input():
    return "This is "

def test_openai_api():
    inputs = raw_input()
    result = ''
    '''
    result = utils.openaiAPI.chatgpt_request(
            inputs,
            model = "gpt-3.5-turbo",
            system_role = "You are a helpful assistant.",
            num_retries = 3,
            waiting_time = 1)
    '''
    assert result is not None
