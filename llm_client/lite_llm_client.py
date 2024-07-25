from typing import List
from llm_client.anthropic_client import _AnthropicClient, AnthropicConfig
from llm_client.config import _LLMConfig, OpenAIConfig
from llm_client.interfaces import _LLMClient, LLMMessage
from llm_client.openai_client import _OpenAIClient

class LiteLLMClient():
  config:_LLMConfig
  client:_LLMClient

  def __init__(self, config:_LLMConfig):
    self.config = config

    if isinstance(config, OpenAIConfig):
      self.client = _OpenAIClient(config)
    elif isinstance(config, AnthropicConfig):
      self.client = _AnthropicClient(config)
    

  def chat_completions(self, messages:List[LLMMessage]):
    return self.client.chat_completions(messages=messages)