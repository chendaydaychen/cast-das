"""Live and replayable LLM decisions for the CreditReview workload.

Only an explicit live mode makes network requests.  The original synthetic
workload remains the default.  No API key or response body is logged here.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from threading import BoundedSemaphore
from urllib.parse import urlparse
from typing import Any


@dataclasses.dataclass(frozen=True)
class ReasoningResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    response_id: str = ""
    admission_wait_ms: float = 0.0


class OpenAIResponsesReasoner:
    def __init__(self, model: str, *, api_key: str | None = None,
                 endpoint: str = "https://api.openai.com/v1/responses", timeout_s: float = 120,
                 allow_insecure_http: bool = False, api_style: str = "responses",
                 instructions: str | None = None, expose_phase_in_prompt: bool = True):
        if not model:
            raise ValueError("an explicit model snapshot/name is required")
        scheme = urlparse(endpoint).scheme
        if scheme != "https" and not (allow_insecure_http and scheme == "http"):
            raise ValueError("a HTTPS endpoint is required; plain HTTP needs explicit opt-in")
        if api_style not in {"responses", "chat"}:
            raise ValueError("api_style must be responses or chat")
        self.model = model
        self._key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._key:
            raise ValueError("OPENAI_API_KEY is required for live mode")
        self.endpoint = endpoint
        self.api_style = api_style
        self.timeout_s = timeout_s
        if instructions is not None and not instructions.strip():
            raise ValueError("custom instructions must be nonempty")
        self.instructions = instructions
        self.expose_phase_in_prompt = bool(expose_phase_in_prompt)

    def reason(self, *, label: str, phase: str, context: str) -> ReasoningResult:
        instructions = self.instructions or (
                "You are an enterprise credit-review agent. For evidence-route, "
                "reply with exactly the risk category low, medium, or high shown "
                "in the profile. Otherwise give a short review note."
            )
        prompt = ((f"Phase: {phase}\n" if self.expose_phase_in_prompt else "")
                  + f"Step: {label}\nObserved data: {context}")
        if self.api_style == "responses":
            body = dict(model=self.model, store=False, instructions=instructions, input=prompt)
        else:
            body = dict(model=self.model, store=False, messages=[
                dict(role="developer", content=instructions),
                dict(role="user", content=prompt),
            ])
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=payload, headers={
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }, method="POST")
        for retry in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or retry == 3:
                    raise
                try:
                    suggested = float((exc.headers or {}).get("Retry-After", "0"))
                except (TypeError, ValueError):
                    suggested = 0.0
                time.sleep(min(15.0, max(suggested, float(2 ** retry))))
        if self.api_style == "responses":
            texts = [part.get("text", "") for item in data.get("output", [])
                     if item.get("type") == "message"
                     for part in item.get("content", []) if part.get("type") == "output_text"]
        else:
            texts = [choice.get("message", {}).get("content") or ""
                     for choice in data.get("choices", [])]
        usage = data.get("usage") or {}
        return ReasoningResult("\n".join(texts).strip(),
                               int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
                               int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
                               str(data.get("model") or self.model), str(data.get("id") or ""))


class LimitedReasoner:
    """Bound concurrent API calls without changing transaction worker count."""

    def __init__(self, delegate: Any, semaphore: BoundedSemaphore):
        self.delegate, self.semaphore = delegate, semaphore

    def reason(self, *, label: str, phase: str, context: str) -> ReasoningResult:
        started = time.perf_counter()
        with self.semaphore:
            waited_ms = (time.perf_counter() - started) * 1000
            result = self.delegate.reason(label=label, phase=phase, context=context)
        return dataclasses.replace(result, admission_wait_ms=waited_ms)


class ReplayReasoner:
    def __init__(self, records: list[dict[str, Any]]):
        self._records = iter(records)

    def reason(self, *, label: str, phase: str, context: str) -> ReasoningResult:
        record = next(self._records)
        if (record["label"], record["phase"], record["context"]) != (label, phase, context):
            raise ValueError("replay prompt diverged from captured CreditReview trace")
        delay_ms = float(record["elapsed_ms"])
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)
        return ReasoningResult(**record["result"])


class CapturingReasoner:
    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.records: list[dict[str, Any]] = []

    def reason(self, *, label: str, phase: str, context: str) -> ReasoningResult:
        started = time.perf_counter()
        result = self.delegate.reason(label=label, phase=phase, context=context)
        elapsed_ms = (time.perf_counter()-started)*1000
        self.records.append(dict(label=label, phase=phase, context=context,
                                 elapsed_ms=elapsed_ms,
                                 model_service_elapsed_ms=max(0.0,
                                     elapsed_ms - result.admission_wait_ms),
                                 result=dataclasses.asdict(result)))
        return result
