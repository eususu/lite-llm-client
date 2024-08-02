# lite llm client

This project made for very light llm client.
the main idea is `do not use llm client library`.

# setup

## How to pass `API_KEY`

1. use parameter of LLMConfig
```python
LiteLLMClient(OpenAIConfig(api_key="YOUR API KEY"))
```
2. use .env
    - rename `.env_example` to `.env`
    - replace YOUR KEY to real api_key

```bash
OPENAI_API_KEY=YOUR KEY
ANTHROPIC_API_KEY=YOUR KEY
GEMINI_API_KEY=YOUR KEY
```


# Known issue

- gemini path may not stable. guide code has `/v1beta/...`. sometimes gemini returns http 500 error

# Roadmap

- [x] `2024-07-21` support OpenAI
- [x] `2024-07-25` support Anthropic
- [x] `2024-07-27` add options for inference
- [x] `2024-07-28` support Gemini
- [x] `2024-07-30` support streaming (OpenAI). simple SSE implement.
- [x] `2024-07-31` support streaming (Anthropic).
- [x] `2024-08-01` support streaming (Gemini). unstable google gemini.
- [ ] support multimodal (image and text)


# Reference

- [Gemini REST API](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=rest)