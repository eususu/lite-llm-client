from enum import Enum
from typing import Optional
from pydantic import BaseModel


class SupportedModel(Enum):
  GPT_4_O = "gpt-4o"

class LLMConfig(BaseModel):
  pass

class OpenAIConfig(LLMConfig):
  model:SupportedModel