from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from enum import Enum
from typing import Optional
from pydantic import BaseModel

# load .env
load_dotenv()

class OpenAIModel(Enum):
  GPT_4_O = "gpt-4o"

@dataclass
class LLMConfig(ABC):
  @abstractmethod
  def get_chat_completion_url(self)->str:
    raise NotImplementedError

###################################################
class OpenAIConfig(LLMConfig):
  base_url: str
  api_key: str
  chat_completion_path: Optional[str] ="/v1/chat/completions"
  model:OpenAIModel|str

  def __init__(self,
               base_url:str="https://api.openai.com",
               api_key:str=None,
               model:OpenAIModel|str=OpenAIModel.GPT_4_O):
    """
    parameters
    - api_key: if None, use environment variable "OPENAI_API_KEY"
    """

    self.base_url = base_url
    self.api_key = api_key
    if not self.api_key and "OPENAI_API_KEY" in os.environ:
      self.api_key = os.environ["OPENAI_API_KEY"]
    self.model = model
  
  def get_chat_completion_url(self)->str:
    return f'{self.base_url}{self.chat_completion_path}'

###################################################
class AnthropicModel(Enum):
  CLAUDE_3_5_SONNET_20240620="claude-3-5-sonnet-20240620"

class AnthropicConfig(LLMConfig):
  base_url: str
  api_key: str
  chat_completion_path: Optional[str] ="/v1/messages"
  model:AnthropicModel|str

  max_tokens:int=1024

  def __init__(self,
               base_url:str="https://api.anthropic.com",
               api_key:str=None,
               model:AnthropicModel|str=AnthropicModel.CLAUDE_3_5_SONNET_20240620):
    """
    parameters
    - api_key: if None, use environment variable "ANTHROPIC_API_KEY"
    """

    self.base_url = base_url
    self.api_key = api_key
    if not self.api_key and "ANTHROPIC_API_KEY" in os.environ:
      self.api_key = os.environ["ANTHROPIC_API_KEY"]
    self.model = model

  def get_chat_completion_url(self)->str:
    return f'{self.base_url}{self.chat_completion_path}'