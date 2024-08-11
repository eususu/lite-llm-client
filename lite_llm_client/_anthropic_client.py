import logging
from typing import Iterator, List

import requests
from lite_llm_client._config import AnthropicConfig
from lite_llm_client._http_sse import SSEDataType, decode_sse
from lite_llm_client._interfaces import InferenceOptions, LLMClient, LLMMessage, LLMMessageRole


class AnthropicClient(LLMClient):
  config: AnthropicConfig
  
  def __init__(self, config:AnthropicConfig):
    self.config = config

  def _make_and_send_request(self, messages:List[LLMMessage], options:InferenceOptions, use_sse=False)->requests.Response:
    _options = options if options else InferenceOptions()
    msgs = []
    system_prompt = []
    for msg in messages:
      role = None
      if msg.role == LLMMessageRole.USER:
        role = "user"
      elif msg.role == LLMMessageRole.SYSTEM:
        system_prompt.append(msg.content)
        continue
      elif msg.role == LLMMessageRole.ASSISTANT:
        role = "assistant"
      else:
        logging.fatal("unknown role")

      msgs.append({"role": role, "content": msg.content})

    """
    https://docs.anthropic.com/en/api/messages
    
    system_prompt does not include messages.
    """
    request = {
      "model": self.config.model.value,
      'max_tokens': self.config.max_tokens,
      "messages": msgs,
      "temperature": _options.temperature,
    }

    if len(system_prompt) > 0:
      request['system'] = "\n".join(system_prompt)

    if _options.top_k:
      request['top_k'] = _options.top_k
    if _options.top_p:
      request['top_p'] = _options.top_p

    if use_sse:
      request['stream'] = True

    http_response = requests.api.post(
      self.config.get_chat_completion_url(),
      headers={
        'Content-Type': 'application/json',
        'x-api-key': f'{self.config.api_key}',
        'anthropic-version': '2023-06-01',
        },
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
      event_value example:
      {"type":"message_start","message":{"id":"msg_01C4SDTbBPX6yQiFSrgnY8jD","type":"message","role":"assistant","model":"claude-3-opus-20240229","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":13,"output_tokens":3}}             }
      """
      event_type = event.event_value['type']
      if event_type == 'message_start':
        # just start signal
        continue

      if event_type == 'content_block_start':
        # ok start
        continue

      if event_type == 'ping':
        # what the ping
        continue

      if event_type == 'content_block_stop':
        continue
      if event_type == 'message_delta':
        """
        {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}              }
        """
        continue
      if event_type == 'message_stop':
        continue

      if event_type == 'content_block_delta':
        """delta example:
        {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello! How"} }
        """
        delta = event.event_value['delta']
        char = delta['text']
        yield char

  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions):
    http_response = self._make_and_send_request(messages=messages, options=options)
    response = http_response.json()

    content = response['content'][0]
    return content['text']