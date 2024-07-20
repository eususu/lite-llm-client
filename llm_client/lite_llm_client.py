from llm_client.anthropic_client import AnthropicClient, AnthropicConfig
from llm_client.config import LLMConfig, OpenAIConfig
from llm_client.interfaces import LLMClient
from llm_client.openai_client import OpenAIClient


class LiteLLMClient():
  config:LLMConfig
  client:LLMClient

  def __init__(self, config:LLMConfig):
    self.config = config

    if isinstance(config, OpenAIConfig):
      self.client = OpenAIClient(config)
    elif isinstance(config, AnthropicConfig):
      self.client = AnthropicClient(config)
    

  def chat_completions(self, messages):
    return self.client.chat_completions(messages=messages)