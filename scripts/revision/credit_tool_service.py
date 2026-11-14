#!/usr/bin/env python3
"""Local-only HTTP test service, not an external production credit provider."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class VerificationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/verify":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 4096:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            if not {"company_id", "task_seed", "risk", "score"} <= request.keys():
                raise ValueError("missing verification fields")
            if request["risk"] not in {"low", "medium", "high"}:
                raise ValueError("unknown risk class")
            if 'evidence_contract' in request:
                from agent.workloads.credit_evidence_feedback import verification_result
                result = verification_result(request)
            else:
                canonical = json.dumps(request, sort_keys=True).encode("utf-8")
                result = {"verified": True,
                                  "receipt_id": hashlib.sha256(canonical).hexdigest()[:16],
                                  "service": "local-test-verifier"}
            payload = json.dumps(result).encode('utf-8')
        except (ValueError, TypeError, KeyError):
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def make_server(port=0):
    return ThreadingHTTPServer(("127.0.0.1", port), VerificationHandler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=35201)
    args = parser.parse_args()
    service = make_server(args.port)
    print(f"local-test-verifier: http://127.0.0.1:{service.server_port}/v1/verify", flush=True)
    service.serve_forever()
