"""Test the evaluation function."""

import sys
sys.path.insert(0, '../src')
import pytest
from datasets import load_dataset
from rageval import evaluate
from rageval.metrics import ContextRecall, AnswerNLIGroundedness
from langchain.llms.fake import FakeListLLM
from rageval.models import NLIModel


#@pytest.mark.slow
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

    ds = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"] 
    ds = ds.rename_column("question", "questions")
    ds = ds.rename_column("answer", "answers")
    ds = ds.rename_column("ground_truths", "gt_answers")
    # crop answers longer than 300 words, since tiny nli model has maximum sequence length of 500
    def truncate_answer(example):
        max_length = 100
        contexts = []
        for context in example["contexts"]:
            contexts.append([c[:max_length] if len(c) > max_length else c for c in context])
        example["contexts"] = contexts
        return example
    ds = ds.map(truncate_answer, batched=True)
    # define model for each metric
    cr_model = FakeListLLM(responses=['[\n    {\n        "statement_1":"恐龙的命名始于1841年，由英国科学家理查德·欧文命名。",\n        "reason": "The answer provides the exact year and the scientist who named the dinosaurs.",\n        "Attributed": "1"\n    },\n    {\n        "statement_2":"欧文在研究几块样子像蜥蜴骨头化石时，认为它们是某种史前动物留下来的，并命名为恐龙。",\n        "reason": "The answer accurately describes the process of how dinosaurs were named.",\n        "Attributed": "1"\n    }\n]'])
    ag_model = NLIModel('text-classification', 'hf-internal-testing/tiny-random-RobertaPreLayerNormForSequenceClassification')
    # run evaluate
    result, instance_level_result = evaluate(
        ds.select(range(3)),
        metrics=[ContextRecall(), AnswerNLIGroundedness()],
        models = [cr_model, ag_model]
    )
    assert result is not None
    assert instance_level_result is not None

#test_evaluation()
