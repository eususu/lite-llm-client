from dotenv import load_dotenv
from enum import Enum
from typing import Optional
from pydantic import BaseModel

# load .env
load_dotenv()


class SupportedModel(Enum):
  GPT_4_O = "gpt-4o"

class LLMConfig(BaseModel):
  pass

class OpenAIConfig(LLMConfig):
  base_url: Optional[str]="https://api.openai.com"
  api_key: Optional[str]=None,
  chat_completion_path: Optional[str] ="/v1/chat/completions"
  model:SupportedModel
  
  def get_chat_completion_url(self)->str:
    return f'{self.base_url}{self.chat_completion_path}'

class AnthropicModel(Enum):
  A="123"

class AnthropicConfig(LLMConfig):
  model:AnthropicModel