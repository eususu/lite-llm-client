import json
from typing import List

from lite_llm_client._types import _ITracer

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace import get_current_span, Span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

otlp_exporter = OTLPSpanExporter(
    #endpoint="http://eususu.synology.me:4317",
    endpoint="http://172.16.10.108:4317",
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

from openinference.semconv.trace import SpanAttributes, MessageAttributes, MessageContentAttributes
from openinference.semconv.trace import OpenInferenceSpanKindValues
class _OtelTracer(_ITracer):


  def start_as_current_span(self, name):
    return tracer.start_as_current_span(name)

  def add_llm_info(self, llm_provider:str, model_name:str, messages:List[dict], extra_args:dict):
    span = get_current_span()

    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
    span.set_attribute(SpanAttributes.LLM_PROVIDER, llm_provider)
    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)

    for index, m in enumerate(messages):
      span.set_attribute(f'{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_ROLE}', m["role"])
      span.set_attribute(f'{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_CONTENT}', m["content"])

    span.set_attribute(SpanAttributes.INPUT_VALUE, messages[-1]["content"])

    span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(extra_args))

  def add_llm_output(self, output:str):
    span = get_current_span()
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, output)
    span.set_status(status=Status(StatusCode.OK))

  def add_llm_usage(self, prompt_tokens:int, completion_tokens:int, total_tokens:int):
    span = get_current_span()
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens)


