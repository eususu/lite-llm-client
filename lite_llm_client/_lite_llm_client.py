from typing import List
from lite_llm_client._anthropic_client import AnthropicClient
from lite_llm_client._config import LLMConfig, OpenAIConfig, AnthropicConfig
from lite_llm_client._interfaces import LLMClient, LLMMessage
from lite_llm_client._openai_client import OpenAIClient

class LiteLLMClient():
  config:LLMConfig
  client:LLMClient

  def __init__(self, config:LLMConfig):
    self.config = config

    if isinstance(config, OpenAIConfig):
      self.client = OpenAIClient(config)
    elif isinstance(config, AnthropicConfig):
      self.client = AnthropicClient(config)
    

  def chat_completions(self, messages:List[LLMMessage]):
    return self.client.chat_completions(messages=messages)