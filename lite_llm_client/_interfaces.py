from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
import logging
from typing import Iterator, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, field_validator, validator

@dataclass
class LLMConfig(ABC):
  @abstractmethod
  def get_chat_completion_url(self)->str:
    raise NotImplementedError

class LLMMessageRole(IntEnum):
  USER=1
  SYSTEM=2
  ASSISTANT=3

  def to_str(self):
    if self.value == 1:
      return "user"
    elif self.value == 2:
      return "system"
    elif self.value == 3:
      return "assistant"

class LLMMessage(BaseModel):
  role: LLMMessageRole
  content: str

class InferenceResult(BaseModel):
  finish_reason:Literal['empty', 'stop']='empty'
  prompt_tokens:int=0
  completion_tokens:int=0
  total_tokens:int=0

class InferenceOptions(BaseModel):
  top_p:Optional[float]=None
  top_k:Optional[float]=None
  max_tokens: Optional[int]=None
  temperature:float=0.0
  inference_result:InferenceResult=InferenceResult()
  batch_mode:bool=False


class LLMResponse(BaseModel):
  text:str
  request:Optional[dict]=None

  def __str__(self)->str:
    return self.text

class LLMClient(ABC):
  @abstractmethod
  def chat_completions(self, messages:List[LLMMessage], options:InferenceOptions)->LLMResponse:
    raise NotImplementedError

  @abstractmethod
  def async_chat_completions(self, messages:List[LLMMessage], options:InferenceOptions)->Iterator[str]:
    raise NotImplementedError


class LLMFileInfo(BaseModel):
  model_config = ConfigDict(extra="allow")

  id:str
  filename:str

class LLMFiles(ABC):

  def create_jsonl(self, file_name:str, jsonl=List[dict])->LLMFileInfo:
    import json
    data = []
    for line in jsonl:
      dump = json.dumps(line, ensure_ascii=False)
      data.append(dump)
    return self.create(file_name=file_name, file_data="\n".join(data))

  @abstractmethod
  def create(self, file_name:str, file_data:str)->LLMFileInfo:
    raise NotImplementedError

  @abstractmethod
  def content(self, file_id:str)->str:
    raise NotImplementedError

  @abstractmethod
  def list()->List[dict]:
    raise NotImplementedError

  @abstractmethod
  def delete(file_id:str):
    raise NotImplementedError

from datetime import datetime

class LLMBatchInfo(BaseModel):
  model_config = ConfigDict(extra="allow")

  id:str
  input_file_id:str
  created_at:datetime
  in_progress_at:Optional[datetime]
  expires_at:Optional[datetime]
  finalizing_at:Optional[datetime]
  completed_at:Optional[datetime]
  request_counts:dict = {"total": 0, "completed": 0, "failed": 0}

  @field_validator('created_at', 'in_progress_at', 'expires_at', 'finalizing_at', 'completed_at', mode="before")
  def convert_timestamp(cls, v):
    try:
      dt = datetime.fromtimestamp(v)
      return dt
    except:
      return None



class LLMBatch(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def create(self, file_id:str)->LLMBatchInfo:
    raise NotImplementedError

  @abstractmethod
  def cancel(self, batch_id:str):
    raise NotImplementedError

  @abstractmethod
  def list()->List[LLMBatchInfo]:
    raise NotImplementedError