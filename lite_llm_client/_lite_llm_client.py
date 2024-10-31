import logging
from typing import Iterator, List
from lite_llm_client._anthropic_client import AnthropicClient
from lite_llm_client._config import GeminiConfig, LLMConfig, OpenAIConfig, AnthropicConfig
from lite_llm_client._gemini_client import GeminiClient
from lite_llm_client._interfaces import InferenceOptions, LLMClient, LLMMessage, LLMMessageRole
from lite_llm_client._openai_client import OpenAIClient

from lite_llm_client._tracer import tracer

@tracer.start_as_current_span("LiteLLMClient")
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

  @tracer.start_as_current_span("constructor")
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
    
  @tracer.start_as_current_span("chat_completions with query")
  def chat_completion(self, query:str, options:InferenceOptions=InferenceOptions()):
    messages:List[LLMMessage]= []
    messages.append(LLMMessage(role=LLMMessageRole.USER, content=query))

    return self.chat_completions(messages=messages, options=options)

  @tracer.start_as_current_span("chat_completions")
  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions=InferenceOptions()):
    r"""chat completions
    
    :param messages: messages
    :param options: (optional) options for chat completions
    :return answer of LLM

    """

    if options is None:
      options = InferenceOptions()
    return self.client.chat_completions(messages=messages, options=options)

  @tracer.start_as_current_span("async_chat_completions")
  def async_chat_completions(self, messages:List[LLMMessage], options:InferenceOptions=InferenceOptions())->Iterator[str]:
    r"""chat completions
    
    :param messages: messages
    :param options: (optional) options for chat completions
    :return parts of answer. use generator

    """
    if options is None:
      options = InferenceOptions()
    return self.client.async_chat_completions(messages=messages, options=options)