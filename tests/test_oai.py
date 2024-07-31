import os
import sys
from typing import List, Union

sys.path.append(os.path.abspath('.'))
import logging
from _share import get_test_messages
from lite_llm_client._config import OpenAIConfig, OpenAIModel
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole
from lite_llm_client._lite_llm_client import LiteLLMClient


def gen_instance()->LiteLLMClient:
  client = LiteLLMClient(OpenAIConfig(
    model=OpenAIModel.GPT_4_O
    ))

  return client

def test_oai_sync():
  client = gen_instance()

  answer = client.chat_completions(messages=get_test_messages())
  logging.info("{}".format(answer))


def test_oai_async():
  client = gen_instance()

  answer = client.async_chat_completions(messages=get_test_messages())
  for a in answer:
    print(a)

