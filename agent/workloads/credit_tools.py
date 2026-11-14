"""Explicit HTTP tool boundary and deterministic capture/replay for CreditReview."""

from __future__ import annotations

import json
import time
import urllib.request
from urllib.parse import urlparse
from typing import Any


class HttpVerificationTool:
    def __init__(self, endpoint: str, *, timeout_s: float = 20.0,
                 allow_insecure_http: bool = False, allow_evidence_feedback: bool = False):
        scheme = urlparse(endpoint).scheme
        if scheme != "https" and not (scheme == "http" and allow_insecure_http):
            raise ValueError("tool endpoint requires HTTPS or explicit HTTP opt-in")
        self.endpoint, self.timeout_s = endpoint, timeout_s
        self.allow_evidence_feedback = bool(allow_evidence_feedback)

    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(request, sort_keys=True).encode("utf-8")
        message = urllib.request.Request(
            self.endpoint, data=body, headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(message, timeout=self.timeout_s) as response:
            result = json.load(response)
        if self.allow_evidence_feedback:
            from .credit_evidence_feedback import is_evidence_feedback
            if is_evidence_feedback(result):
                return result
        if result.get("verified") is not True or not result.get("receipt_id"):
            raise ValueError("credit verification tool did not acknowledge request")
        return result


class CapturingTool:
    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.records: list[dict[str, Any]] = []

    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.delegate.apply(request)
        self.records.append(dict(request=request, response=result,
                                 elapsed_ms=(time.perf_counter()-started)*1000))
        return result


class ReplayTool:
    def __init__(self, records: list[dict[str, Any]]):
        self._records = iter(records)

    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        record = next(self._records)
        if record["request"] != request:
            raise ValueError("replayed tool request diverged from captured task")
        if record["elapsed_ms"] > 0:
            time.sleep(record["elapsed_ms"] / 1000)
        return record["response"]
