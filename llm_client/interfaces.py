from enum import IntEnum
from typing import List
from pydantic import BaseModel

class LLMMessageRole(IntEnum):
  USER=1
  SYSTEM=2
  ASSISTANT=3

class LLMMessage(BaseModel):
  role: LLMMessageRole
  content: str

class LLMClient:
  def chat_completions(self, messages:List[LLMMessage]):
    pass