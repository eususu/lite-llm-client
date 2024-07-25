import logging
from llm_client._config import OpenAIConfig, OpenAIModel
from llm_client._interfaces import LLMMessage, LLMMessageRole
from llm_client._lite_llm_client import LiteLLMClient

client = LiteLLMClient(OpenAIConfig(
  model=OpenAIModel.GPT_4_O
  ))

messages = [
  LLMMessage(role=LLMMessageRole.USER, content="hello")
]

answer = client.chat_completions(messages=messages)

logging.info("asdf {}".format(answer))

