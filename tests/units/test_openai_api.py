# -*- coding: utf-8 -*-

import os
import pytest

from langchain.llms.fake import FakeListLLM

from rageval.models import OpenAILLM


@pytest.fixture(scope='module')
def test_case():
    questions = ["恐龙是怎么被命名的？"]
    ground_truths = [["1841年，英国科学家理查德·欧文在研究几块样子像蜥蜴骨头化石时，认为它们是某种史前动物留下来的，并命名为恐龙，意思是“恐怖的蜥蜴”。"]]
    answers = ["人从恐龙进化而来"]
    contexts = [["[12]恐龙是 介于冷血和温血之间的动物2014年6月，有关恐龙究竟是像鸟类和哺乳动物一样的温血动物，还是类似爬行动物、鱼类和两栖动物的冷血动物的问题终于有了答案——恐龙其实是介于冷血和温血之间的动物。 [12]“我们的结果显示恐龙所具有的生长速率和新陈代谢速率，既不是冷血生物体也不是温血生物体所具有的特征。它们既不像哺乳动物或者鸟类，也不像爬行动物或者鱼类，而是介于现代冷血动物和温血动物之间。简言之，它们的生理机能在现代社会并不常见。”美国亚利桑那大学进化生物学家和生态学家布莱恩·恩奎斯特说。墨西哥生物学家表示，正是这种中等程度的新陈代谢使得恐龙可以长得比任何哺乳动物都要大。温血动物需要大量进食，因此它们频繁猎捕和咀嚼植物。“很难想象霸王龙大小的狮子能够吃饱以 存活下来。","[12]哺乳动物起源于爬行动物，它们的前身是“似哺乳类的爬行动物”，即兽孔目，早期则是“似爬行类的哺乳动物”，即哺乳型动物。 [12]中生代的爬行动物，大部分在中生代的末期灭绝了；一部分适应了变化的环境被保留下来，即现存的爬行动物（如龟鳖类、蛇类、鳄类等）；还有一部分沿着不同的进化方向，进化成了现今的鸟类和哺乳类。 [12]恐龙是 介于冷血和温血之间的动物2014年6月，有关恐龙究竟是像鸟类和哺乳动物一样的温血动物，还是类似爬行动物、鱼类和两栖动物的冷血动物的问题终于有了答案——恐龙其实是介于冷血和温血之间的动物。"]]
    return {'questions': questions,
            'ground_truths': ground_truths,
            'answers': answers,
            'contexts': contexts}

@pytest.mark.skip
def test_openai_api(test_case):
    os.environ["OPENAI_API_KEY"] = "NOKEY"
    client = OpenAILLM("gpt-3.5-turbo-16k", "OPENAI_API_KEY")

    # test request
    results = client.generate(test_case['questions'])
    assert results is not None

def test_fakelistllm_api(test_case):
    model = FakeListLLM(responses=["I'll callback later.", "You 'console' them!"])

    # test request
    results = model.generate(test_case['questions'])
    assert results is not None
