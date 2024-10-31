from lite_llm_client._types import _ITracer

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True
)

resource = Resource(attributes={
  SERVICE_NAME: "lite-llm-client"
})

# 트레이서 프로바이더 설정
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

class _OtelTracer(_ITracer):

  def start_as_current_span(self, name):
    return tracer.start_as_current_span(name)
