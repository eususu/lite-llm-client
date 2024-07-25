from dotenv import load_dotenv
from enum import Enum
from typing import Optional
from pydantic import BaseModel

# load .env
load_dotenv()

class OpenAIModel(Enum):
  GPT_4_O = "gpt-4o"

class LLMConfig(BaseModel):
  pass

###################################################
class OpenAIConfig(LLMConfig):
  base_url: Optional[str]="https://api.openai.com"
  api_key: Optional[str]=None,
  chat_completion_path: Optional[str] ="/v1/chat/completions"
  model:OpenAIModel|str
  
  def get_chat_completion_url(self)->str:
    return f'{self.base_url}{self.chat_completion_path}'

###################################################
class AnthropicModel(Enum):
  CLAUDE_3_5_SONNET_20240620="claude-3-5-sonnet-20240620"

class AnthropicConfig(LLMConfig):
  base_url: Optional[str]="https://api.anthropic.com"
  api_key: Optional[str]=None,
  chat_completion_path: Optional[str] ="/v1/messages"
  model:AnthropicModel|str

  max_tokens:int=1024

  def get_chat_completion_url(self)->str:
    return f'{self.base_url}{self.chat_completion_path}'