import pytest
from datasets import Dataset

from rageval.metrics import TextLength
import rageval as rl


#@pytest.fixture(scope='module')
def sample():
    test_case = {
        #"questions": [
        #    "习近平主席在何时何地会见了埃塞俄比亚总理海尔马里亚姆？",
        #    "埃塞俄比亚希望与中国在哪些领域加强合作？"
        #],
        "answers": [
            "习近平主席在2017年5月12日于北京人民大会堂会见了埃塞俄比亚总理海尔马里亚姆。",
            "埃塞俄比亚希望与中国在以下领域加强合作：\n1. **共建“一带一路”框架下合作**：埃塞俄比亚表示希望能够积极参与“一带一路”倡议，深化与中国在基础设施建设、产能合作、互联互通等领域的合作。\n2. **提高工业化水平和出口创汇能力**：埃塞俄比亚期待中国在推动其工业化进程以及提升出口创汇能力方面提供帮助和合作。\n3. **安全、有序、有效推进经贸合作**：希望与中方在贸易和投资合作领域取得进展，实现稳定、有序和高效的合作。"
        ]
    }
    return test_case


#@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


#@pytest.mark.slow
def test_case_on_text_length(testset):
    tokenize_model = rl.models.Tokenizer("Qwen/Qwen2-0.5B-Instruct")
    metric = TextLength(tokenize_model=tokenize_model)
    assert metric.name == "text_length"
    score, results = metric.compute(testset["answers"], batch_size = 1)
    print(score, results)
    assert score == 75.0

test_case_on_text_length(testset(sample()))