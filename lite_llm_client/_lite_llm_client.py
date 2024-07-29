from typing import Iterator, List
from lite_llm_client._anthropic_client import AnthropicClient
from lite_llm_client._config import GeminiConfig, LLMConfig, OpenAIConfig, AnthropicConfig
from lite_llm_client._gemini_client import GeminiClient
from lite_llm_client._interfaces import InferenceOptions, LLMClient, LLMMessage
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
    elif isinstance(config, GeminiConfig):
      self.client = GeminiClient(config)

      
    #if not self.client:
    #  raise NotImplementedError()
    

  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions=None):
    r"""chat completions
    
    :param messages: messages
    :param options: (optional) options for chat completions

    """
    return self.client.chat_completions(messages=messages, options=options)

  def async_chat_completions(self, messages:List[LLMMessage], options:InferenceOptions=None)->Iterator[str]:
    r"""chat completions
    
    :param messages: messages
    :param options: (optional) options for chat completions
    :return answer: parts of answer. use generator

    """
    return self.client.async_chat_completions(messages=messages, options=options)