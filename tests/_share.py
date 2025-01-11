from typing import List
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole


def get_test_messages()->List[LLMMessage]:
  messages = [
    LLMMessage(role=LLMMessageRole.SYSTEM, content="you are helpful assistant."),
    LLMMessage(role=LLMMessageRole.USER, content="hello. can you tell me about this lite-llm-client project? if you need the information, you can get it on 'https://github.com/eususu/lite-llm-client'")
  ]

  return messages

def get_json_test_messages()->List[LLMMessage]:
  messages = [
    LLMMessage(role=LLMMessageRole.SYSTEM, content="Extract the event information."),
    LLMMessage(role=LLMMessageRole.USER, content="Alice and Bob are going to a science fair on Friday.")
  ]

  return messages

def get_json_schema()->dict:
  """
  "json_schema": {
        "name": "research_paper_extraction",
        "schema": {
          "type": "object",
          "properties": {
            "title": { "type": "string" },
            "authors": {
              "type": "array",
              "items": { "type": "string" }
            },
            "abstract": { "type": "string" },
            "keywords": {
              "type": "array",
              "items": { "type": "string" }
            }
          },
          "required": ["title", "authors", "abstract", "keywords"],
          "additionalProperties": false
        },
        "strict": true
      }
  """
  schema = {
    "name": "calendar_event_extraction",
    "schema": {
      "type": "object",
      "properties": {
        "event": { "type": "string" },
        "participants": {
          "type": "array",
          "items": { "type": "string" }
        },
        "date": { "type": "string" },
        "location": { "type": "string" }
      },
      "required": ["event", "participants", "date", "location"],
      "additionalProperties": False
    },
    "strict": True
  }

  return schema