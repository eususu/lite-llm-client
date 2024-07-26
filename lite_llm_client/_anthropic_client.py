import logging
from typing import List

import requests
from lite_llm_client._config import AnthropicConfig
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole


class AnthropicClient():
  config: AnthropicConfig
  
  def __init__(self, config:AnthropicConfig):
    self.config = config

  def chat_completions(self, messages:List[LLMMessage]):
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
      'max_tokens': self.config.max_tokens,
      "messages": msgs,
      "temperature": 0.0
    }

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