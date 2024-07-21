import logging
from llm_client.config import OpenAIConfig, SupportedModel
from llm_client.interfaces import LLMMessage, LLMMessageRole
from llm_client.lite_llm_client import LiteLLMClient

client = LiteLLMClient(OpenAIConfig(
  model=SupportedModel.GPT_4_O
  ))

messages = [
  LLMMessage(role=LLMMessageRole.USER, content="hello")
]

answer = client.chat_completions(messages=messages)

logging.info("asdf {}".format(answer))

