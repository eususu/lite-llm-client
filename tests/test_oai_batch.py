import os
import sys
import json

sys.path.append(os.path.abspath('.'))
import logging
from _share import get_json_schema, get_json_test_messages, get_test_messages
from lite_llm_client import OpenAIConfig, OpenAIModel
from lite_llm_client import LiteLLMClient
from lite_llm_client import InferenceOptions

def gen_instance()->LiteLLMClient:
  client = LiteLLMClient(OpenAIConfig(
    model=OpenAIModel.GPT_4O_MINI
    ))

  return client

def test_oai_batch_sync():
  client = gen_instance()
  options = InferenceOptions(batch_mode=True)

  queries = ["tell me lite llm client project.", "tell me github"]
  contexts = [None, None]
  filename = "batch_request_1.jsonl"
  file_info = None

  file_list = client.files.list()
  for file in file_list:
    logging.info(file)
    if file.filename == filename:
      file_info = file
      break
  """

      client.files.delete(file.id)
  batch_request = []
  for index, (query, context) in enumerate(zip(queries, contexts)):
    res = client.chat_completion(query, context, system_prompt='you are helpful assistant', options=options)

    res.request["custom_id"] = f'request-id-{index}'
    batch_request.append(res.request)

  file_info = client.files.create_jsonl("batch_request_1.jsonl", batch_request)
  logging.info(file_info.id)
  #client.files.delete(file_info.id)

  file_list = client.files.list()
  for file in file_list:
    logging.info(file)


  batch_info = client.batch.create(file_id=file_info.id)
  logging.info(batch_info)
  """

  batch_list = client.batch.list()
  for batch in batch_list:
    if batch.input_file_id == file_info.id:
      logging.info(batch)

      if batch.completed_at:
        logging.info(f'{batch.request_counts} 배치 작업 완료에 걸린시간: {batch.completed_at - batch.created_at}')
      break

