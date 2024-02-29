# -*- coding: utf-8 -*-

from typing import Union, List

from rageval.metrics import Metric
from rageval.tasks import BaseTask


class Generate(BaseTask):
    name = 'Generator'
    # Define required columns in testset for the evaluation of the task
    required_columns = ['questions', 'answers', 'gt_answers']

    def __init__(self, metrics: Union[str, List[str], List[Metric]]):
        """Init task"""

        super().__init__(metrics)
