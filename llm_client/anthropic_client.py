from typing import List
from llm_client.config import AnthropicConfig
from llm_client.interfaces import LLMMessage


class AnthropicClient():
  config: AnthropicConfig
  
  def __init__(self, config:AnthropicConfig):
    self.config = config

  def chat_completions(self, messages:List[LLMMessage]):
    return f"hi anthropic my model is {self.config.model}"