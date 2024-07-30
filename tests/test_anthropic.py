import os
import sys
sys.path.append(os.path.abspath('.'))
import logging
from lite_llm_client._anthropic_client import AnthropicConfig 
from lite_llm_client._config import AnthropicModel
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole
from lite_llm_client._lite_llm_client import LiteLLMClient

logging.basicConfig(level='debug')
def test_oai():
  client = LiteLLMClient(AnthropicConfig(model=AnthropicModel.CLAUDE_3_OPUS_20240229))

  messages = [
    LLMMessage(role=LLMMessageRole.SYSTEM, content="you are helpful assistant."),
    LLMMessage(role=LLMMessageRole.USER, content="hello")
  ]

  answer = client.async_chat_completions(messages=messages)
  for a in answer:
    logging.info(a)

  answer = client.chat_completions(messages=messages)
  logging.info("{}".format(answer))

