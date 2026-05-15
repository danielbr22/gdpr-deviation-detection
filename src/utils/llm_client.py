import os
import sys
import asyncio
import random

import httpx

_OPENAI_KEY: str = os.environ.get("OPENAI_API_KEY", "")
# Auto-select openai when OPENAI_API_KEY is present, unless overridden explicitly.
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai" if _OPENAI_KEY else "ollama")
_OLLAMA_BASE: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
_OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_URL: str = "https://api.openai.com/v1/chat/completions"

_DEFAULT_CONCURRENCY = {"ollama": 1, "openai": 5}


def concurrency() -> int:
    default = _DEFAULT_CONCURRENCY.get(LLM_PROVIDER, 1)
    return int(os.environ.get("LLM_CONCURRENCY", default))


async def call_async(system: str, user: str, *, json_mode: bool = False, timeout: int = 120) -> str:
    if LLM_PROVIDER == "openai":
        return await _openai(system, user, json_mode=json_mode, timeout=timeout)
    return await _ollama(system, user, json_mode=json_mode, timeout=timeout)


async def _ollama(system: str, user: str, *, json_mode: bool, timeout: int) -> str:
    payload: dict = {
        "model": _OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "/no_think\n" + user},
        ],
    }
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(_OLLAMA_BASE + "/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]


async def _openai(system: str, user: str, *, json_mode: bool, timeout: int) -> str:
    headers = {
        "Authorization": f"Bearer {_OPENAI_KEY}",
        "content-type": "application/json",
    }
    payload: dict = {
        "model": _OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for attempt in range(6):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(_OPENAI_URL, json=payload, headers=headers)
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt == 5:
                raise
            await asyncio.sleep(2 ** attempt + random.random())
            continue
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2 ** attempt + random.random()))
            await asyncio.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    r.raise_for_status()
    return ""  # unreachable


def check_provider() -> None:
    if LLM_PROVIDER == "openai":
        if not _OPENAI_KEY:
            sys.exit("ERROR: OPENAI_API_KEY not set")
        try:
            httpx.get("https://api.openai.com", timeout=5)
        except Exception as e:
            sys.exit(f"ERROR: Cannot reach OpenAI API — {e}")
        print(f"  Provider : OpenAI")
        print(f"  Model    : {_OPENAI_MODEL}")
        print(f"  Concurrency: {concurrency()}")
    else:
        try:
            r = httpx.get(_OLLAMA_BASE + "/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
        except Exception as e:
            sys.exit(f"ERROR: Ollama unreachable at {_OLLAMA_BASE} — {e}")
        if not any(_OLLAMA_MODEL in m for m in models):
            sys.exit(f"ERROR: {_OLLAMA_MODEL} not found — run: ollama pull {_OLLAMA_MODEL}")
        print(f"  Provider : Ollama at {_OLLAMA_BASE}")
        print(f"  Model    : {_OLLAMA_MODEL}")
        print(f"  Models   : {', '.join(models)}")
        print(f"  Concurrency: {concurrency()}")
