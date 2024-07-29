import logging
from typing import Iterator, List
from lite_llm_client._config import OpenAIConfig
import requests

from lite_llm_client._interfaces import InferenceOptions, LLMMessage, LLMMessageRole
from lite_llm_client._http_sse import SSEDataType, decode_sse

class OpenAIClient():
  config:OpenAIConfig

  def __init__(self, config:OpenAIConfig):
    self.config = config

    
  def _make_and_send_request(self, messages:List[LLMMessage], options:InferenceOptions, use_sse=False)->requests.Response:
    _options = options if options else InferenceOptions()
    msgs = []
    for msg in messages:
      role = None
      if msg.role == LLMMessageRole.USER:
        role = "user"
      elif msg.role == LLMMessageRole.SYSTEM:
        role = "system"
      elif msg.role == LLMMessageRole.ASSISTANT:
        role = "assistant"
      else:
        logging.fatal("unknown role")

      msgs.append({"role": role, "content": msg.content})

    request = {
      "model": self.config.model.value,
      "messages": msgs,
      "temperature": _options.temperature,
    }

    if use_sse:
      request['stream'] = True

    #logging.info(f'request={request}')

    http_response = requests.api.post(
      self.config.get_chat_completion_url(),
      headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.config.api_key}'},
      json=request
      )

    if http_response.status_code != 200:
      logging.fatal(f'response={http_response.text}')
      raise Exception(f'bad status_code: {http_response.status_code}')

    return http_response

  def async_chat_completions(self, messages:List[LLMMessage], options:InferenceOptions)->Iterator[str]:
    http_response = self._make_and_send_request(messages=messages, options=options, use_sse=True)

    for event in decode_sse(http_response, data_type=SSEDataType.JSON):
      """
      value example:
      {'id': 'chatcmpl-9qLv6AAbMZcZudyYUJ2SsSYGZs16y', 'object': 'chat.completion.chunk', 'created': 1722264344, 'model': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_400f27fa1f', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'logprobs': None, 'finish_reason': None}]}
      """
      delta = event.event_value['choices'][0]['delta']
      if 'content' in delta:
        char = delta['content']
        logging.debug(char)
        yield char
        # CHECK: first token is empty. is useful??
      else:
        # maybe last data?
        pass


  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions):
    http_response = self._make_and_send_request(messages=messages, options=options)
    response = http_response.json()
    #logging.info(f'response={response}')

    choices = response['choices']
    return choices[0]["message"]["content"]