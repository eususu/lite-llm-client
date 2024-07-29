from enum import Enum, auto
import json
import logging
from typing import Iterator
from pydantic import BaseModel
from requests import Response

class SSEEvent(BaseModel):
  event_name:str
  event_value:str|dict

class SSEDataType(Enum):
  TEXT=auto()
  JSON=auto()

def _parse_sse(line:bytes):
  first_comma = line.find(b': ', 0)
  if first_comma == -1:
    raise ValueError('sse parse error (1)')

  name = line[:first_comma] 
  value = line[first_comma+2:]
  return name.decode(),value.decode()


def decode_sse(response:Response, data_type:SSEDataType, eoe:str='[DONE]')->Iterator[SSEEvent]:
  ct = response.headers.get('Content-Type') # Content-Type: text/event-stream; utf-8
  ct_values = ct.split(';')
  assert ct_values[0] == 'text/event-stream', "response content-type does not 'text/event-stream '"

  logging.info(f'[Response] Content-Type: {ct}')
  for line in response.iter_lines(delimiter=b'\n\n'):
    parsed_line = _parse_sse(line)

    if parsed_line[0] == 'data' and parsed_line[1] == eoe:
      logging.info("END OF EVENT")
      break
    #logging.info(f'response line={parsed_line}')

    value:str
    if SSEDataType.JSON == data_type:
      value = json.loads(parsed_line[1])
    #elif SSEDataType.TEXT == data_type:
    else:
      value = parsed_line[1]

    yield SSEEvent(event_name=parsed_line[0], event_value=value)
  