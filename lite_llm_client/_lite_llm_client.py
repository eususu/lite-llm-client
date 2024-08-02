from typing import Iterator, List
from lite_llm_client._anthropic_client import AnthropicClient
from lite_llm_client._config import GeminiConfig, LLMConfig, OpenAIConfig, AnthropicConfig
from lite_llm_client._gemini_client import GeminiClient
from lite_llm_client._interfaces import InferenceOptions, LLMClient, LLMMessage
from lite_llm_client._openai_client import OpenAIClient

class LiteLLMClient():
  """
  This is lite-llm-client class.
  it supports three types of client

  OpenAI usage:

  >>> from lite_llm_client import LiteLLMClient, OpenAIConfig
  >>> client = LiteLLMClient(OpenAIConfig(api_key="your api key"))

  Gemini usage:

  >>> from lite_llm_client import LiteLLMClient, GeminiConfig
  >>> client = LiteLLMClient(GeminiConfig(api_key="your api key"))

  Anthropic usage:

  >>> from lite_llm_client import LiteLLMClient, AnthropicConfig
  >>> client = LiteLLMClient(AnthropicConfig(api_key="your api key"))
  """
  config:LLMConfig
  client:LLMClient=None

  def __init__(self, config:LLMConfig):
    self.config = config

    if isinstance(config, OpenAIConfig):
      self.client = OpenAIClient(config)
    elif isinstance(config, AnthropicConfig):
      self.client = AnthropicClient(config)
    elif isinstance(config, GeminiConfig):
      self.client = GeminiClient(config)

    if not self.client:
      raise NotImplementedError()
    

  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions=None)->str:
    r"""chat completions function
    
    :param messages: messages
    :param options: (optional) options for chat completions
    :return answer of LLM

    """
    return self.client.chat_completions(messages=messages, options=options)

  def async_chat_completions(self, messages:List[LLMMessage], options:InferenceOptions=None)->Iterator[str]:
    r"""chat completions
    
    :param messages: messages
    :param options: (optional) options for chat completions
    :return parts of answer. use generator

    """
    return self.client.async_chat_completions(messages=messages, options=options)