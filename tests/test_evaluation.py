"""Test the evaluation function."""

import sys

import pytest
from datasets import load_dataset, Dataset
from langchain.llms.fake import FakeListLLM

import rageval as rl
from rageval.models import NLIModel

sys.path.insert(0, '../src')


@pytest.mark.slow
def test_evaluation():
    """
    This is test unit for testing the load_dataset function.
    """

    # 1) init test task: task type, metrics

    # 2) load dataset, and extract testset
    # train_data = rageval.datasets.load_data('', task='')
    # assert len(train_data) == 300

    # 3) run evaluation
    # result = evaluate(testset, info)

    ds = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"] 
    ds = ds.rename_column("question", "questions")
    ds = ds.rename_column("answer", "answers")
    ds = ds.rename_column("ground_truths", "gt_answers")

    # crop answers longer than 300 words, since tiny nli model has maximum sequence length of 500
    def truncate_answer(example):
        max_length = 100
        answers = []
        gt_answers = []
        """
        for a in example["answers"]:
            answers.append([c[:max_length] if len(c) > max_length else c for c in a])
        example["answers"] = answers
        """
        for ga in example["gt_answers"]:
            gt_answers.append([q[:max_length] if len(q) > max_length else q for q in ga])
        example["gt_answers"] = gt_answers
        return example
    ds = ds.map(truncate_answer, batched=True)

    # define model for each metric
    cr_model = FakeListLLM(
        responses=[
            '[\n    {\n        "statement_1":"恐龙的命名始于1841年，由英国科学家理查德·欧文命名。",\n        "reason": "The answer provides '
            'the exact year and the scientist who named the dinosaurs.",\n        "Attributed": "1"\n    },\n    {\n'
            '        "statement_2":"欧文在研究几块样子像蜥蜴骨头化石时，认为它们是某种史前动物留下来的，并命名为恐龙。",\n        "reason": "The answer '
            'accurately describes the process of how dinosaurs were named.",\n        "Attributed": "1"\n    }\n]'
        ]
    )
    ag_model = NLIModel(
        'text-classification',
        'hf-internal-testing/tiny-random-RobertaPreLayerNormForSequenceClassification'
    )

    # define metrics
    metrics = [
        rl.metrics.ContextRecall(cr_model),
        rl.metrics.AnswerNLICorrectness(nli_model=ag_model, decompose_model="nltk")
    ]

    # define task
    task = rl.tasks.Generate(metrics=metrics)

    # run evaluate
    result = task.evaluate(ds)
    assert isinstance(result, dict)
    detailed_result = task.obtain_detailed_result()
    assert isinstance(detailed_result, Dataset)
