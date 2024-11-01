import os
import sys
from typing import List

sys.path.append(os.path.abspath('.'))
import logging
from _share import get_test_messages
from lite_llm_client._config import OpenAIConfig
from lite_llm_client._interfaces import LLMMessage, LLMMessageRole
from lite_llm_client._lite_llm_client import LiteLLMClient
from lite_llm_client._tracer import tracer


client = OpenAIConfig()
llc = LiteLLMClient(client)
#answer = llc.chat_completion('hi', context="친절하게 답해줘", system_prompt="you are helpful assistant.")
#logging.info(answer)

from opentelemetry.trace import Span, get_current_span
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues, EmbeddingAttributes, DocumentAttributes

def llm(query:str):
    messages = []
    messages.append(LLMMessage(role=LLMMessageRole.SYSTEM, content="you are helpful assistant"))
    messages.append(LLMMessage(role=LLMMessageRole.USER, content="나는 세개의 계란을 가지고 있어."))
    messages.append(LLMMessage(role=LLMMessageRole.ASSISTANT, content="네, 당신은 세개의 계란을 가지고 있습니다."))
    messages.append(LLMMessage(role=LLMMessageRole.USER, content=query))
    answer = llc.chat_completions(messages=messages)
    logging.info(answer)

@tracer.start_as_current_span(__name__)
def _embedding(query:str, docs:List[str]):
    span = get_current_span()
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
    span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, "bge-m3")

    vecs = []
    for doc in docs:
        vecs.append([1.231, 1.331, 0.8, -0.77])

    for index, vec in enumerate(vecs):
        span.set_attribute(f'{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}.{EmbeddingAttributes.EMBEDDING_TEXT}', docs[index])
        span.set_attribute(f'{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}.{EmbeddingAttributes.EMBEDDING_VECTOR}', vec)

@tracer.start_as_current_span(__name__)
def _retriever(query:str):
    span = get_current_span()
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)

    docs = ['관련문서1', '관련문서2']
    span.set_attribute(SpanAttributes.INPUT_VALUE, query)

    for index, doc in enumerate(docs):
        span.set_attribute(f'{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}.{DocumentAttributes.DOCUMENT_ID}', index)
        span.set_attribute(f'{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}.{DocumentAttributes.DOCUMENT_CONTENT}', doc)
        span.set_attribute(f'{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}.{DocumentAttributes.DOCUMENT_METADATA}', '{"METADATA": "test", "src":"/mnt/d/test"}')
        span.set_attribute(f'{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}.{DocumentAttributes.DOCUMENT_SCORE}', 0.8217)

    _embedding(query, docs)


@tracer.start_as_current_span(__name__)
def test_chain():
    span = get_current_span()
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)

    query = "여기에서 내가 두개를 먹으면 몇개가 남지?"
    _retriever(query)

    #llm()