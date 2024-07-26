import os
import sys
sys.path.append(os.path.abspath('.'))
import logging
from lite_llm_client._config import OpenAIConfig, OpenAIModel
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole
from lite_llm_client._lite_llm_client import LiteLLMClient

def test_oai():
  client = LiteLLMClient(OpenAIConfig(
    model=OpenAIModel.GPT_4_O
    ))

  messages = [
    LLMMessage(role=LLMMessageRole.USER, content="hello")
  ]

  answer = client.chat_completions(messages=messages)

  logging.info("{}".format(answer))

