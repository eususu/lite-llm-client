import os
import sys
from typing import List, Union

sys.path.append(os.path.abspath('.'))
import logging
from _share import get_test_messages
from lite_llm_client._config import OpenAIConfig
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole
from lite_llm_client._lite_llm_client import LiteLLMClient


client = OpenAIConfig()
llc = LiteLLMClient(client)
answer = llc.chat_completion('hi')
logging.info(answer)