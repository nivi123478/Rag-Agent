---
title: Document Q&A Agent
emoji: 📄
sdk: gradio
app_file: app.py
pinned: false
---

# Document Q&A (RAG) Agent: free to run and host

Upload PDFs or text files and ask questions. Flow: rewrite question → hybrid retrieval (embeddings + BM25) →
relevance grading → one retry with a new query → cited answer (or "not found").

## Run locally

```bash
pip install -r requirements.txt
export LLM_BASE_URL=...   # see presets below
export LLM_API_KEY=...
export LLM_MODEL=...
python app.py
```

## Free LLM presets (OpenAI-compatible)

Free-tier limits and model names change often, so check the provider's current docs.

| Provider | LLM_BASE_URL | Example LLM_MODEL |
|---|---|---|
| Groq | https://api.groq.com/openai/v1 | llama-3.3-70b-versatile |
| Gemini | https://generativelanguage.googleapis.com/v1beta/openai/ | a current Flash model from AI Studio |

Optional failover when the first provider hits its rate limit: set `FALLBACK_LLM_BASE_URL`,
`FALLBACK_LLM_API_KEY`, `FALLBACK_LLM_MODEL`.

## Deploy free on Hugging Face Spaces

1. Create a new Space, choose the **Gradio** SDK and the free CPU hardware.
2. Upload all files in this folder (the front matter above configures the Space).
3. Under Settings → Variables and secrets, add `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`
   (add the key as a *secret*).
4. The first build is slow because it installs sentence-transformers.

Notes: indexes live in memory per session and are lost on restart. Free LLM tiers may train on prompts,
so use non-sensitive documents only.

## Evaluate

Edit `eval_set.json`, then run `python eval.py eval_set.json your.pdf`.
