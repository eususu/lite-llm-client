from enum import Enum
import logging
from typing import Iterator, List
from lite_llm_client._config import OpenAIConfig
import requests

from lite_llm_client._interfaces import InferenceOptions, InferenceResult, LLMBatch, LLMBatchInfo, LLMClient, LLMFileInfo, LLMFiles, LLMMessage, LLMMessageRole, LLMResponse
from lite_llm_client._http_sse import SSEDataType, decode_sse
from lite_llm_client._tracer import tracer

def _send_request(base_url:str, path:str, headers:dict, method:str="POST", request:dict=None, files:dict=None)->requests.Response:
  args = {
    "url":f'{base_url}/{path}',
    "headers":headers,
  }

  if request:
    args["json"]=request

  if files:
    args["files"] = files

  if method == "GET":
    http_response = requests.api.get(**args)
  elif method == "DELETE":
    http_response = requests.api.delete(**args)
  elif method == "POST":
    http_response = requests.api.post(**args)
  else:
    raise NotImplementedError

  if http_response.status_code != 200:
    logging.fatal(f'response={http_response.text}')
    raise Exception(f'bad status_code: {http_response.status_code}')

  return http_response

class OpenAIClient(LLMClient):
  config:OpenAIConfig

  def __init__(self, config:OpenAIConfig):
    self.config = config
    
  def _make_request(self, messages:List[LLMMessage], json_schema:dict|None, options:InferenceOptions, use_sse=False)->requests.Response:
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

    model_name = self.config.model.value if isinstance(self.config.model, Enum) else self.config.model
    request = {
      "model": model_name,
      "temperature": _options.temperature,
    }

    if use_sse:
      request['stream'] = True

    tracer.add_llm_info(llm_provider="OpenAI", model_name=model_name, messages=msgs, extra_args=request)

    request["messages"] = msgs
    if json_schema:
      request["response_format"] = {
        "type": "json_schema",
        "json_schema": json_schema
      }
    return request

  
  def _parse_response(self, inference_options:InferenceOptions, response:dict):
    choices0 = response['choices'][0]

    if 'usage' in response:
      usage = response['usage']
      prompt_tokens = usage['prompt_tokens']
      completion_tokens = usage['completion_tokens']
      total_tokens = usage['total_tokens']

      tracer.add_llm_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)

      if inference_options:
        if inference_options.inference_result:
          inference_result = inference_options.inference_result
          inference_result.finish_reason = choices0['finish_reason']
          inference_result.prompt_tokens = prompt_tokens
          inference_result.completion_tokens = completion_tokens
          inference_result.total_tokens = total_tokens

          logging.info(inference_result)

  def _parse_async_response(self, inference_result:InferenceResult, response:dict)->str:
    choices0 = response['choices'][0]
    finish_reason = choices0['finish_reason']
    if finish_reason is not None:
      inference_result.finish_reason = finish_reason

    delta = choices0['delta']
    if 'content' in delta:
      char = delta['content']
      inference_result.completion_tokens += len(char)
      inference_result.total_tokens += len(char)
      return char
    return None

  def async_chat_completions(self, messages:List[LLMMessage], options:InferenceOptions, json_schema:dict=None)->Iterator[str]:
    # TODO: count prompt length
    
    http_request = self._make_request(messages=messages, json_schema=json_schema, options=options)
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {self.config.api_key}'
    }
    http_response = _send_request(
      base_url=self.config.base_url,
      path=self.config.chat_completion_path,
      headers=headers,
      request=http_request)

    for event in decode_sse(http_response, data_type=SSEDataType.JSON):
      """
      value example:
      {'id': 'chatcmpl-9qLv6AAbMZcZudyYUJ2SsSYGZs16y', 'object': 'chat.completion.chunk', 'created': 1722264344, 'model': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_400f27fa1f', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'logprobs': None, 'finish_reason': None}]}
      """
      char = self._parse_async_response(inference_result=options.inference_result, response=event.event_value)

      if char:
        yield char
      else:
        # may be last?
        pass


  def chat_completions(self, messages:List[LLMMessage], json_schema:dict, options:InferenceOptions)->LLMResponse:

    if options.batch_mode:
      request = self._make_request(messages=messages, json_schema=json_schema, options=options)
      return LLMResponse(text="", request={
        "method": "POST",
        "url": self.config.chat_completion_path,
        "body":request
        }
      )

    http_request = self._make_request(messages=messages, json_schema=json_schema, options=options)
    
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {self.config.api_key}'
    }
    http_response = _send_request(
      base_url=self.config.base_url,
      path=self.config.chat_completion_path,
      headers=headers,
      request=http_request)
    response = http_response.json()

    self._parse_response(options, response)

    choices = response['choices']
    return LLMResponse(text=choices[0]["message"]["content"])



####################################
class OpenAIFilesClient(LLMFiles):
  config:OpenAIConfig
  def __init__(self, config:OpenAIConfig):
    self.config = config

  def create(self, file_name:str, file_data:str)->LLMFileInfo:
    logging.info(file_name)
    logging.info(file_data)

    headers = {
      'Authorization': f'Bearer {self.config.api_key}'
    }

    files_request = {
      "purpose": (None, 'batch'),
      "file": (file_name, file_data),
    }
    http_response = _send_request(base_url=self.config.base_url, path=self.config.files_path, headers=headers, files=files_request)
    res = http_response.json()
    return LLMFileInfo(**res)

  def content(self, file_id:str)->str:
    headers = {
      'Authorization': f'Bearer {self.config.api_key}'
    }

    http_response = _send_request(base_url=self.config.base_url, path=f'{self.config.files_path}/{file_id}/content', headers=headers, method="GET")
    res = http_response.text
    return res

  def list(self)->List[LLMFileInfo]:
    headers = {
      'Authorization': f'Bearer {self.config.api_key}'
    }

    http_response = _send_request(base_url=self.config.base_url, path=self.config.files_path, headers=headers, method="GET")
    res = http_response.json()

    file_list = []
    for file in res['data']:
      file_list.append(LLMFileInfo(**file))

    return file_list

  def delete(self, file_id:str):
    headers = {
      'Authorization': f'Bearer {self.config.api_key}'
    }

    http_response = _send_request(base_url=self.config.base_url, path=f'{self.config.files_path}/{file_id}', headers=headers, method="DELETE")
    res = http_response.json()

    return res

class OpenAIBatchClient(LLMBatch):
  config:OpenAIConfig
  def __init__(self, config:OpenAIConfig):
    self.config = config

  def create(self, file_id:str)->LLMBatchInfo:
    endpoint = "/v1/chat/completions"
    batch_request = {
      "input_file_id": file_id,
      "endpoint": endpoint,
      "completion_window": "24h"
    }
    
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {self.config.api_key}'
    }
    http_response = _send_request(base_url=self.config.base_url, path=self.config.batch_path, headers=headers, request=batch_request)

    res = http_response.json()
    return LLMBatchInfo(**res)

  def cancel(self, batch_id:str)->LLMBatchInfo:
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {self.config.api_key}'
    }
    http_response = _send_request(base_url=self.config.base_url, path=f'{self.config.batch_path}/{batch_id}/cancel', headers=headers, method="POST")

    res = http_response.json()
    return LLMBatchInfo(**res)

  def list(self)->List[LLMBatchInfo]:
    headers = {
      'Authorization': f'Bearer {self.config.api_key}'
    }

    http_response = _send_request(base_url=self.config.base_url, path=self.config.batch_path, headers=headers, method="GET")
    res = http_response.json()


    batch_infos = []
    for batch in res['data']:
      batch_infos.append(LLMBatchInfo(**batch))

    return batch_infos