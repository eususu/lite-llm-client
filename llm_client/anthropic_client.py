from enum import Enum
from llm_client.config import LLMConfig


class AnthropicModel(Enum):
  A="123"

class AnthropicConfig(LLMConfig):
  model:AnthropicModel
class AnthropicClient():
  config: AnthropicConfig
  
  def __init__(self, config:AnthropicConfig):
    self.config = config

  def chat_completions(self, messages):
    return f"hi anthropic my model is {self.config.model}"