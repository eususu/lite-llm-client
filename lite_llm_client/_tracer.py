import importlib
import logging
from typing import List

from lite_llm_client._types import _ITracer


class _DummyTracer(_ITracer):
  """
  open telemetry가 활성화 되지 않았을때 아무것도 안할 tracer
  """
  def start_as_current_span(self, name):
    def decorator(func):
      def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
      return wrapper
    return decorator

  def add_llm_info(self, llm_provider:str, model_name:str, messages:List[dict], extra_args:dict):
    pass

  def add_llm_output(self, output:str):
    pass

  def add_llm_usage(self, prompt_tokens:int, completion_tokens:int, total_tokens:int):
    pass


tracer:_ITracer=None

try:
    # opentelemetry 모듈이 로딩 가능하면 _Tracer 객체 생성
    importlib.import_module('opentelemetry')
    from lite_llm_client._otel_tracer import _OtelTracer
    tracer = _OtelTracer()

except ImportError as e:
    # opentelemetry 모듈이 없으면 _DummyTracer 객체 생성
    logging.warning(e)
    tracer = _DummyTracer()

logging.info(f"Loaded tracer={tracer}")