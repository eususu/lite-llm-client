import os
import sys
from typing import List, Union
sys.path.append(os.path.abspath('.'))
import logging
from lite_llm_client._anthropic_client import AnthropicConfig 
from lite_llm_client._config import AnthropicModel
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole
from lite_llm_client._lite_llm_client import LiteLLMClient

logging.basicConfig(level='debug')

def gen_instance()->Union[LiteLLMClient, List[LLMMessage]]:
  client = LiteLLMClient(AnthropicConfig(model=AnthropicModel.CLAUDE_3_OPUS_20240229))

  messages = [
    LLMMessage(role=LLMMessageRole.SYSTEM, content="you are helpful assistant."),
    LLMMessage(role=LLMMessageRole.USER, content="hello")
  ]

  return client, messages

def test_anthropic_sync():
  client, messages = gen_instance()

  answer = client.chat_completions(messages=messages)
  logging.info("{}".format(answer))


def test_anthropic_async():
  client, messages = gen_instance()

  answer = client.async_chat_completions(messages=messages)
  for a in answer:
    logging.info(a)

