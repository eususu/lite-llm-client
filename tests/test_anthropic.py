import logging
from llm_client.anthropic_client import AnthropicConfig, AnthropicModel
from llm_client.lite_llm_client import LiteLLMClient

logging.basicConfig(level='debug')

client = LiteLLMClient(AnthropicConfig(model=AnthropicModel.A))

answer = client.chat_completions(messages="hello")

logging.info("asdf {}".format(answer))

