from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional
from pydantic import BaseModel

@dataclass
class LLMConfig(ABC):
  @abstractmethod
  def get_chat_completion_url(self)->str:
    raise NotImplementedError

class LLMMessageRole(IntEnum):
  USER=1
  SYSTEM=2
  ASSISTANT=3

class LLMMessage(BaseModel):
  role: LLMMessageRole
  content: str

class InferenceOptions(BaseModel):
  top_p:Optional[float]=None
  top_k:Optional[float]=None
  max_tokens: Optional[int]=None
  temperature:float=0.0

class LLMClient(ABC):
  @abstractmethod
  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions=InferenceOptions()):
    raise NotImplementedError