import logging
from llm_client.config import OpenAIConfig, SupportedModel
from llm_client.lite_llm_client import LiteLLMClient

logging.basicConfig(level='debug')

client = LiteLLMClient(OpenAIConfig(
  model=SupportedModel.GPT_4_O
  ))

messages = [
  {"role": "user", "content":"hello"}
]

answer = client.chat_completions(messages=messages)

logging.info("asdf {}".format(answer))

