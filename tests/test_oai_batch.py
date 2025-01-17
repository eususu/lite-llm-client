import os
import sys

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


  batch_request = []
  for (query, context) in zip(queries, contexts):
    res = client.chat_completion(query, context, system_prompt='you are helpful assistant', options=options)
    batch_request.append(res.request)

  client.batch.create(batch_request)

