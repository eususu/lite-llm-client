import os
import sys
from typing import List, Union

sys.path.append(os.path.abspath('.'))
import logging
from _share import get_test_messages
from lite_llm_client._anthropic_client import AnthropicConfig 
from lite_llm_client._config import AnthropicModel
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole
from lite_llm_client._lite_llm_client import LiteLLMClient

logging.basicConfig(level='debug')

def gen_instance()->LiteLLMClient:
  client = LiteLLMClient(AnthropicConfig(model=AnthropicModel.CLAUDE_3_OPUS_20240229))

  return client

def test_anthropic_sync():
  client = gen_instance()

  answer = client.chat_completions(messages=get_test_messages())
  logging.info("{}".format(answer))


def test_anthropic_async():
  client = gen_instance()

  answer = client.async_chat_completions(messages=get_test_messages())
  for a in answer:
    logging.info(a)

