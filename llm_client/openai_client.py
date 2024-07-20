from llm_client.config import OpenAIConfig

class OpenAIClient():
  config:OpenAIConfig

  def __init__(self, config:OpenAIConfig):
    self.config = config

  def chat_completions(self, messages):
    return "hi oai"