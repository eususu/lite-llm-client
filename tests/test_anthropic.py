import logging
from llm_client._anthropic_client import AnthropicConfig 
from llm_client._config import AnthropicModel
from llm_client._interfaces import LLMMessage, LLMMessageRole
from llm_client._lite_llm_client import LiteLLMClient

logging.basicConfig(level='debug')

client = LiteLLMClient(AnthropicConfig(model=AnthropicModel.CLAUDE_3_5_SONNET_20240620))

messages = [
  LLMMessage(role=LLMMessageRole.USER, content="hello")
]

answer = client.chat_completions(messages=messages)

logging.info("{}".format(answer))

