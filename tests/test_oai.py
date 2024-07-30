import os
import sys
from typing import List, Union
sys.path.append(os.path.abspath('.'))
import logging
from lite_llm_client._config import OpenAIConfig, OpenAIModel
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole
from lite_llm_client._lite_llm_client import LiteLLMClient


def gen_instance()->Union[LiteLLMClient, List[LLMMessage]]:
  client = LiteLLMClient(OpenAIConfig(
    model=OpenAIModel.GPT_4_O
    ))

  messages = [
    LLMMessage(role=LLMMessageRole.USER, content="hello")
  ]

  return client, messages

def test_oai_sync():
  client, messages = gen_instance()

  answer = client.chat_completions(messages=messages)
  logging.info("{}".format(answer))


def test_oai_async():
  client, messages = gen_instance()

  answer = client.async_chat_completions(messages=messages)
  for a in answer:
    print(a)

