import os
import sys
from typing import List, Union
sys.path.append(os.path.abspath('.'))
import logging
from lite_llm_client._config import GeminiConfig, GeminiModel
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole
from lite_llm_client._lite_llm_client import LiteLLMClient

def gen_instance()->Union[LiteLLMClient, List[LLMMessage]]:
  client = LiteLLMClient(GeminiConfig(
    model=GeminiModel.GEMINI_1_5_FLASH
    ))

  messages = [
    LLMMessage(role=LLMMessageRole.USER, content="hello")
  ]

  return client, messages

def test_gemini():
  client, messages = gen_instance()

  answer = client.chat_completions(messages=messages)

  logging.info("{}".format(answer))


