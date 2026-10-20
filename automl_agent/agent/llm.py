"""Minimal OpenAI-compatible chat client with token/call budgeting and JSON extraction.

Uses only the standard library. If no endpoint is configured, or the budget is spent,
or a call fails, `ask_json` returns None and the agent falls back to built-in hypotheses.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request

log = logging.getLogger("agent.llm")


def extract_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


class LLM:
    def __init__(self, base_url, api_key, model, token_budget, max_calls, timeout=120.0):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model = model
        self.token_budget = token_budget
        self.max_calls = max_calls
        self.timeout = timeout
        self.tokens_used = 0
        self.calls = 0
        self.failures = 0

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key) and self.calls < self.max_calls \
            and self.tokens_used < self.token_budget and self.failures < 3

    def chat(self, system: str, user: str, max_tokens: int = 1800, temperature: float = 0.4):
        if not self.available:
            return None
        url = self.base_url + ("/chat/completions" if self.base_url.endswith("/v1") or "/v1" in self.base_url
                               else "/v1/chat/completions")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            usage = data.get("usage") or {}
            self.tokens_used += int(usage.get("total_tokens") or (len(system) + len(user)) // 3)
            self.failures = 0
            return data["choices"][0]["message"].get("content") or ""
        except Exception as e:  # network, auth, quota, malformed reply
            self.failures += 1
            log.warning("LLM call failed (%s): %s", self.failures, e)
            return None

    def ask_json(self, system: str, user: str, **kw):
        text = self.chat(system, user, **kw)
        if text is None:
            return None
        obj = extract_json(text)
        if obj is None:
            log.warning("LLM reply is not valid JSON: %.200s", text)
        return obj
