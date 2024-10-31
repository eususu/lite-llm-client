import importlib
import logging

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

tracer:_ITracer=None

try:
    # opentelemetry 모듈이 로딩 가능하면 _Tracer 객체 생성
    importlib.import_module('opentelemetry')
    from lite_llm_client._otel_tracer import _OtelTracer
    tracer = _OtelTracer()
    tracer = _DummyTracer()

except ImportError:
    # opentelemetry 모듈이 없으면 _DummyTracer 객체 생성
    tracer = _DummyTracer()

logging.info(f"Loaded tracer={tracer}")