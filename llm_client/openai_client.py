import os
import logging
from llm_client.config import OpenAIConfig
import requests

class OpenAIClient():
  config:OpenAIConfig

  def __init__(self, config:OpenAIConfig):
    self.config = config

  def chat_completions(self, messages):
    request = {
      "model": self.config.model.value,
      "messages": messages,
      "temperature": 0.0
    }

    logging.info(f'request={request}')

    api_key = self.config.api_key if self.config.api_key is None else os.environ["OPENAI_API_KEY"]

    http_response = requests.api.post(
      self.config.get_chat_completion_url(),
      headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
      json=request
      )

    if http_response.status_code != 200:
      logging.fatal(f'response={http_response.text}')
      raise Exception(f'bad status_code: {http_response.status_code}')
    response = http_response.json()
    logging.info(f'response={response}')

    choices = response['choices']
    return choices[0]["message"]["content"]