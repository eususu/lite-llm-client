import logging
from llm_client.anthropic_client import AnthropicConfig 
from llm_client.config import AnthropicModel
from llm_client.interfaces import LLMMessage, LLMMessageRole
from llm_client.lite_llm_client import LiteLLMClient

logging.basicConfig(level='debug')

client = LiteLLMClient(AnthropicConfig(model=AnthropicModel.A))

messages = [
  LLMMessage(role=LLMMessageRole.USER, content="hello")
]

answer = client.chat_completions(messages=messages)

logging.info("asdf {}".format(answer))

