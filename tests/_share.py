from typing import List
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole


def get_test_messages()->List[LLMMessage]:
  messages = [
    LLMMessage(role=LLMMessageRole.SYSTEM, content="you are helpful assistant."),
    LLMMessage(role=LLMMessageRole.USER, content="hello. can you tell me about this lite-llm-client project? if you need the information, you can get it on 'https://github.com/eususu/lite-llm-client'")
  ]
  

  return messages