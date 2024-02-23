import pytest
from datasets import Dataset
from rageval.metrics import AnswerRouge

class CharTokenizer:
    """Tokenize text into characters."""
    def tokenize(self, text: str) -> list[str]:
        # Tokenize by characters to avoid a dependency on word segmentation methods.
        return [c for c in text]

@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "###刚刚发声，A股这种情况十分罕见！大聪明逆市抄底330亿，一篇研报引爆全球，市场逻辑生变？",
            "The quick brown fox jumps over the lazy dog."
        ],
        "gt_answers": [
            [
                "刚刚过去的这个月，美股总市值暴跌了将近6万亿美元（折合人民币超过40万亿），这背后的原因可能不仅仅是加息这么简单。最近瑞士信贷知名分析师Zoltan Polzsar撰写了一篇极其重要的文章，详细分析了现有世界秩序的崩坏本质以及美国和西方将要采取的应对策略。在该文中，Zoltan Polzsar直指美国通胀的本质和其长期性。同期，A股市场亦出现了大幅杀跌的情况。"
            ],
            [
                "The quick brown fox jumps over the lazy dog.",
                "The brown fox jumps over the lazy dog."
            ]
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


def test_case_on_answer_exact_match(testset):
    metric = AnswerRouge()
    
    # Test with Chinese tokenizer
    chinese_tokenizer = CharTokenizer()
    metric.init_model('rouge1', chinese_tokenizer)
    score, results = metric.compute(testset, 1)
    assert 0 <= score <= 1
    assert isinstance(results, Dataset)
    
    # Test with English tokenizer
    metric.init_model('rouge1')
    score, results = metric.compute(testset, 1)
    assert 0 <= score <= 1
    assert isinstance(results, Dataset)
