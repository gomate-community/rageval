# -*- coding: utf-8 -*-

import os
import pytest
from unittest.mock import patch, MagicMock
import openai
import httpx

from rageval.models import OpenAILLM

from langchain.schema import Generation, LLMResult


@pytest.fixture(scope='module')
def test_case():
    questions = ["截止2001年岛上的人口有多少？", "墨西哥土拨鼠栖息的地区海拔约是多少米？"]
    ground_truths = [
        [
            "总人口9495（2001年）。",
            "墨西哥土拨鼠栖息在海拔1600-2200米的平原上。"
        ]
    ]
    contexts = [
        [
            "米尼科伊岛（Minicoy）位于印度拉克沙群岛中央直辖区最南端，是Lakshadweep县的一个城镇。它与拉克沙群岛隔九度海峡相望，与马尔代夫伊哈万迪富卢环礁隔八度海峡相望。总人口9495（2001年）。米尼科伊岛位于盐水胡东南部的一个有人居住的岛屿，全岛几乎被椰子树覆盖，唯一的地标是一座灯塔。Viringili位于米尼科伊岛西南侧，是一个长度不到两百米的小岛，曾被用作麻风病患者的驱逐地。该地2001年总人口9495人，其中男性4616人，女性4879人；0—6岁人口1129人，其中男571人，女558人；识字率81.95%，其中男性为83.51%，女性为80.47%。",
            "墨西哥土拨鼠（\"Cynomys mexicanus\"），又名墨西哥草原松鼠或墨西哥草原犬鼠，是原住于墨西哥的一种啮齿目。牠们是日间活动的。由于牠们被看为害虫，估其数量下降至濒危水平。墨西哥土拨鼠栖息在海拔1600-2200米的平原上。牠们分布在墨西哥的圣路易斯波托西州北部及科阿韦拉州。牠们主要吃草，并透过食物来吸收水份。牠们有时也会吃昆虫。牠们的天敌有郊狼、短尾猫、鹰、獾及鼬。墨西哥土拨鼠是会冬眠的，其繁殖季节也较短，一般只在1月至4月。妊娠期为1个月，雌鼠每年只会产一胎，一胎平均有四子。幼鼠出生时眼睛闭合，会先以尾巴作为辅助，直至出生后40日才能看见。于5月至6月会断奶，到了1岁就会离开巢穴。冬天前幼鼠就会离开母鼠。幼鼠之间会互咬、嘶叫及扭住来玩耍。牠们1岁后就达至性成熟，寿命约3-5年。成年重约1公斤及长14-17吋，雄性较雌性大只。牠们呈黄色，耳朵较深色，腹部较浅色。墨西哥土拨鼠的语言最为复杂，能奔跑达每小时55公里。所以当受到威胁时，牠们会大叫作为警报，并且高速逃走。墨西哥土拨鼠的巢穴是挖掘出来的。巢穴的入口像漏斗，通道长达100呎，两侧有空间储存食物及休息。巢穴可以多达几百只墨西哥土拨鼠，但一般少于50只，群族有一只雄性的领袖。牠们有时会与斑点黄鼠及穴鸮分享他们的洞穴。于1956年，墨西哥土拨鼠曾在科阿韦拉州、新莱昂州及圣路易斯波托西州出没。到了1980年代，牠们从新莱昂州消失，其分布地少于800平方米。由于牠们被认为是害虫，故经常被毒杀，到了1994年到达濒危的状况。"
        ]
    ]
    return {'questions': questions,
            'ground_truths': ground_truths,
            'contexts': contexts}

@pytest.fixture
def openai_llm():
    return OpenAILLM(model="gpt-3.5-turbo", _api_key_env_var="OPENAI_API_KEY")

def test_init(openai_llm):
    assert openai_llm.model == "gpt-3.5-turbo"
    assert openai_llm.base_url == "https://api.openai.com/v1"
    assert openai_llm.num_retries == 3
    assert openai_llm.timeout == 60
    assert openai_llm.api_key == os.getenv("OPENAI_API_KEY", "NO_KEY")

def test_build_request(openai_llm):
    request = openai_llm.build_request()
    assert request["model"] == "gpt-3.5-turbo"
    assert request["max_tokens"] is None
    assert request["n"] is None
    assert request["temperature"] is None
    assert request["top_p"] is None
    assert request["logprobs"] is None

def test_is_chat_model_engine(openai_llm):
    assert openai_llm._is_chat_model_engine is True

@patch("rageval.models.openai.openai.OpenAI")
def test_llm(mock_openai, openai_llm):
    llm = openai_llm.llm
    mock_openai.assert_called_once_with(
        api_key=openai_llm.api_key,
        base_url=openai_llm.base_url,
        max_retries=openai_llm.num_retries,
        timeout=openai_llm.timeout
    )

@patch("rageval.models.openai.openai.OpenAI")
def test_get_chat_model_response(mock_openai, openai_llm):
    mock_response = MagicMock()
    mock_openai().chat.completions.create.return_value = mock_response
    prompt = [{"role": "user", "content": "Hello"}]
    response = openai_llm._get_chat_model_response(prompt)
    assert response == mock_response
    mock_openai().chat.completions.create.assert_called_once()

@patch("rageval.models.openai.openai.OpenAI")
def test_get_instruct_model_response(mock_openai, openai_llm):
    mock_response = MagicMock()
    mock_openai().completions.create.return_value = mock_response
    prompt = "Hello"
    response = openai_llm._get_instruct_model_response(prompt)
    assert response == mock_response
    mock_openai().completions.create.assert_called_once()

@patch("rageval.models.openai.OpenAILLM._get_chat_model_response")
@patch("rageval.models.openai.OpenAILLM._get_instruct_model_response")
def test_generate(mock_instruct_response, mock_chat_response, openai_llm):
    mock_chat_response.return_value = {"choices": [{"message": {"content": "Hi"}}]}
    mock_instruct_response.return_value = {"choices": [{"text": "Hi"}]}

    prompt_chat = [{"role": "user", "content": "Hello"}]
    result_chat = openai_llm.generate(prompt_chat)
    assert isinstance(result_chat, LLMResult)
    assert result_chat.generations[0][0].text == "Hi"

    prompt_instruct = "Hello"
    openai_llm.model = "gpt-3.5-turbo-instruct"
    result_instruct = openai_llm.generate(prompt_instruct)
    assert isinstance(result_instruct, LLMResult)
    assert result_instruct.generations[0][0].text == "Hi"

def test_create_llm_result(openai_llm):
    response = {
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop", "logprobs": None}]
    }
    result = openai_llm.create_llm_result(response)
    assert isinstance(result, LLMResult)
    assert result.llm_output["token_usage"]["prompt_tokens"] == 10
    assert result.llm_output["token_usage"]["completion_tokens"] == 20
    assert result.llm_output["token_usage"]["total_tokens"] == 30
    assert result.generations[0][0].text == "Hi"

@patch.object(OpenAILLM, 'generate')
def test_batch_generate(mock_generate, openai_llm):
    # Mock the generate method to return a simple LLMResult
    mock_generate.return_value = LLMResult(generations=[[Generation(text="Hi")]])

    # Define prompts for testing
    prompts = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "How are you?"}],
    ]

    # Call batch_generate
    results = openai_llm.batch_generate(prompts, max_workers=2)

    # Verify generate was called with each prompt
    assert mock_generate.call_count == len(prompts)
    mock_generate.assert_any_call(prompts[0])
    mock_generate.assert_any_call(prompts[1])

    # Check results
    assert len(results) == len(prompts)
    for result in results:
        assert isinstance(result, LLMResult)
        assert result.generations[0][0].text == "Hi"

@patch.object(OpenAILLM, 'generate')
def test_batch_generate_order(mock_generate, openai_llm):
    # Mock the generate method to return different results based on input
    def side_effect(prompt):
        if prompt == [{"role": "user", "content": "Hello"}]:
            return LLMResult(generations=[[Generation(text="Hi")]])
        elif prompt == [{"role": "user", "content": "How are you?"}]:
            return LLMResult(generations=[[Generation(text="I am fine")]])
    
    mock_generate.side_effect = side_effect

    # Define prompts for testing
    prompts = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "How are you?"}],
    ]

    # Call batch_generate
    results = openai_llm.batch_generate(prompts, max_workers=2)

    # Check results are in the correct order
    assert results[0].generations[0][0].text == "Hi"
    assert results[1].generations[0][0].text == "I am fine"
