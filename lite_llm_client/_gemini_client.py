import logging
from typing import List

import requests
from lite_llm_client._config import GeminiConfig
from lite_llm_client._interfaces import InferenceOptions, LLMMessage, LLMMessageRole


class GeminiClient():
  config: GeminiConfig
  
  def __init__(self, config:GeminiConfig):
    self.config = config

  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions=InferenceOptions(temperature=1.0, max_tokens=800, top_p=0.8, top_k=10)):
    msgs = []
    system_prompt = []
    for msg in messages:
      role = None
      if msg.role == LLMMessageRole.USER:
        role = "user"
      elif msg.role == LLMMessageRole.SYSTEM:
        role = "system"
        continue
      elif msg.role == LLMMessageRole.ASSISTANT:
        role = "assistant"
      else:
        logging.fatal("unknown role")

      msgs.append({"role": role, "parts": [{'text':msg.content}]})

    
    generationConfig = {
      "temperature": options.temperature,
      "maxOutputTokens": options.max_tokens,
      "topP": options.top_p,
      "topK": options.top_k,
    }
    request = {
      "contents": msgs,
      "generationConfig": generationConfig,
    }

    if len(system_prompt) > 0:
      request['system'] = "\n".join(system_prompt)

    logging.info(f'request={request}')
    http_response = requests.api.post(
      f'{self.config.get_chat_completion_url()}:generateContent?key={self.config.api_key}',
      headers={
        'Content-Type': 'application/json',
        },
      json=request
      )

    if http_response.status_code != 200:
      logging.fatal(f'response={http_response.text}')
      raise Exception(f'bad status_code: {http_response.status_code}')
    response = http_response.json()
    logging.info(f'response={response}')

    content = response['candidates'][0]['content']['parts'][0]
    return content['text']
