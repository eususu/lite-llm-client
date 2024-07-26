import os
import logging
from typing import List
from lite_llm_client._config import OpenAIConfig
import requests

from lite_llm_client._interfaces import InferenceOptions, LLMMessage, LLMMessageRole

class OpenAIClient():
  config:OpenAIConfig

  def __init__(self, config:OpenAIConfig):
    self.config = config

  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions=InferenceOptions()):
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
      "temperature": options.temperature,
    }

    logging.info(f'request={request}')

    http_response = requests.api.post(
      self.config.get_chat_completion_url(),
      headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.config.api_key}'},
      json=request
      )

    if http_response.status_code != 200:
      logging.fatal(f'response={http_response.text}')
      raise Exception(f'bad status_code: {http_response.status_code}')
    response = http_response.json()
    logging.info(f'response={response}')

    choices = response['choices']
    return choices[0]["message"]["content"]