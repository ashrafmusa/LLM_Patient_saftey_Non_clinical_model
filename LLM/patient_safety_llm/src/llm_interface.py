"""Simple LLM interface abstraction for real inference providers.

Resolution order:
1) OpenAI-compatible chat completions API (if API key is configured)
2) Local llama.cpp-compatible HTTP server
3) Local llama binary invocation
4) Safe placeholder fallback
"""
from typing import Dict
import subprocess
import logging

import httpx

from .config import settings

logger = logging.getLogger(__name__)


def generate_response(prompt: str, model: str = 'local') -> Dict:
    """Generate text using a configured real model provider.

    Behavior:
    - If `OPENAI_API_KEY` is set, call OpenAI-compatible `/chat/completions`.
    - Else, if `LLAMA_SERVER_URL` is set, try common generate endpoints.
    - Else, try to call `LLAMA_BINARY` with prompt piped to stdin.
    - On any failure, return a safe placeholder response dict so callers remain functional.
    """
    # 1) Try OpenAI-compatible API first when credentials are available.
    if settings.OPENAI_API_KEY:
        url = settings.OPENAI_BASE_URL.rstrip('/') + '/chat/completions'
        headers = {
            'Authorization': f'Bearer {settings.OPENAI_API_KEY}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': settings.OPENAI_MODEL,
            'messages': [
                {'role': 'system', 'content': 'You are a clinical risk classification assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0,
        }
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=settings.OPENAI_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get('choices', [])
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get('message', {})
                    text = message.get('content', '')
                    if text:
                        return {'text': text, 'model': settings.OPENAI_MODEL, 'meta': data}
            logger.debug('OpenAI-compatible call failed: status=%s body=%s', resp.status_code, resp.text[:500])
        except Exception as e:
            logger.debug('OpenAI-compatible invocation failed: %s', e)

    # 2) Try local HTTP server (recommended for llama.cpp server setups)
    if settings.LLAMA_SERVER_URL:
        # common endpoint patterns used by lightweight LLM servers
        endpoints = ["/generate", "/v1/generate", "/v1/completions"]
        for ep in endpoints:
            url = settings.LLAMA_SERVER_URL.rstrip('/') + ep
            try:
                resp = httpx.post(url, json={"prompt": prompt}, timeout=settings.LLAMA_TIMEOUT)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception:
                        data = {"text": resp.text}
                    # normalize common shapes
                    if isinstance(data, dict) and 'text' in data:
                        return {"text": data.get('text'), "model": "local_server", "meta": data}
                    if isinstance(data, dict) and data.get('choices'):
                        choices = data.get('choices')
                        text = choices[0].get('text') if choices and isinstance(choices[0], dict) else resp.text
                        return {"text": text, "model": "local_server", "meta": data}
                    return {"text": resp.text, "model": "local_server", "meta": data}
            except Exception as e:
                logger.debug("LLM server try failed for %s: %s", url, e)

    # 3) Try calling a local binary (best-effort). This assumes the binary can accept stdin input.
    try:
        bin_path = settings.LLAMA_BINARY
        proc = subprocess.run([bin_path], input=prompt, text=True, capture_output=True, timeout=settings.LLAMA_TIMEOUT)
        out = proc.stdout.strip()
        if out:
            return {"text": out, "model": "local_binary", "meta": {"returncode": proc.returncode}}
    except FileNotFoundError:
        logger.debug("LLAMA binary not found at path: %s", settings.LLAMA_BINARY)
    except Exception as e:
        logger.debug("LLAMA binary invocation failed: %s", e)

    # 4) Fallback placeholder response
    return {"text": "[LLM unavailable — placeholder response]", "model": "fallback", "meta": {}}

