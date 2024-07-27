import logging
from typing import List

import requests
from lite_llm_client._config import AnthropicConfig
from lite_llm_client._interfaces import InferenceOptions, LLMMessage, LLMMessageRole


class AnthropicClient():
  config: AnthropicConfig
  
  def __init__(self, config:AnthropicConfig):
    self.config = config

  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions):
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

    logging.info(f'request={request}')

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
    response = http_response.json()
    logging.info(f'response={response}')

    content = response['content'][0]
    return content['text']