#!/usr/bin/env python3
"""Black-box lifecycle and HTTP contract test for the Swift native service."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import select
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--segment", type=Path)
    parser.add_argument("--bundle", type=Path, action="append", default=[])
    parser.add_argument(
        "--real-control-plane",
        action="store_true",
        help="also load the real model and test overload/cancellation",
    )
    return parser.parse_args()


def request(
    url: str,
    method: str = "GET",
    body: object | bytes | None = None,
    token: str | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict]:
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    value = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(value, timeout=timeout) as response:
            payload = response.read()
            return response.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        payload = error.read()
        return error.code, json.loads(payload) if payload else {}


def ready_line(process: subprocess.Popen[str], timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    assert process.stdout is not None
    while time.monotonic() < deadline:
        readable, _, _ = select.select([process.stdout], [], [], 0.2)
        if readable:
            line = process.stdout.readline()
            if line:
                return json.loads(line)
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"service exited before ready: {process.returncode}: {stderr}")
    raise TimeoutError("service did not emit ready JSON")


def main() -> None:
    args = parse_args()
    if args.segment and args.bundle:
        raise ValueError("--segment and --bundle are mutually exclusive")
    token = "native-contract-test-token"
    command = [
        str(args.binary.resolve()),
        "serve",
        "--package",
        str(args.package.resolve()),
        "--port",
        "0",
        "--auth-token",
        token,
        "--development-deterministic-engine",
        "--skip-warmup",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        ready = ready_line(process)
        assert ready["status"] == "ready"
        base = ready["url"]

        status, payload = request(f"{base}/health")
        assert status == 401 and payload["error"]["type"] == "authentication_error"
        assert payload["error"]["code"] == "authentication_failed"
        status, health = request(f"{base}/health", token=token)
        assert status == 200 and health["status"] == "ready"
        assert health["running_requests"] == 0
        assert health["queued_requests"] == 0
        assert health["max_queued_requests"] == 16
        status, models = request(f"{base}/v1/models", token=token)
        assert status == 200 and len(models["data"]) == 6
        assert models["data"][0]["dimension"] == 2048
        assert models["data"][0]["modalities"] == ["text", "image", "video"]

        status, embedded = request(
            f"{base}/v1/embeddings",
            method="POST",
            token=token,
            body={
                "model": "wemm-embedding-2b-apple-2048",
                "input": "一只坐在窗边的猫",
                "request_id": "contract-smoke",
            },
        )
        vector = embedded["data"][0]["embedding"]
        assert status == 200 and len(vector) == 2048 and vector[0] == 1
        assert embedded["indexed"]["embedding_space"] == "deterministic-test-2048"
        assert embedded["indexed"]["request_id"] == "contract-smoke"

        status, payload = request(
            f"{base}/v1/embeddings",
            method="POST",
            token=token,
            body={"model": "wemm-embedding-2b-apple-12", "input": "bad"},
        )
        assert status == 400 and payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["code"] == "invalid_request"
        status, _ = request(
            f"{base}/v1/embeddings",
            method="POST",
            token=token,
            body=b"{not-json",
        )
        assert status == 400
        status, _ = request(f"{base}/missing", token=token)
        assert status == 404
        status, _ = request(f"{base}/v1/embeddings", method="OPTIONS")
        assert status == 204
        status, health = request(f"{base}/health", token=token)
        assert status == 200
        assert health["requests_total"] == 3
        assert health["requests_succeeded"] == 1
        assert health["requests_failed"] == 2
        assert health["request_latency_ms_average"] >= 0
        assert health["request_latency_ms_last"] >= 0

        process.terminate()
        assert process.wait(timeout=10) == 0
        print(
            json.dumps(
                {
                    "status": "passed",
                    "ready": ready,
                    "checks": 12,
                    "graceful_exit": True,
                },
                ensure_ascii=False,
            )
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    if args.real_control_plane:
        real_control_plane(
            args.binary.resolve(),
            args.package.resolve(),
            args.segment.resolve() if args.segment else None,
            [path.resolve() for path in args.bundle],
        )


def real_control_plane(
    binary: Path,
    package: Path,
    segment: Path | None,
    bundles: list[Path],
) -> None:
    token = "native-real-control-test-token"
    command = [
            str(binary),
            "serve",
            "--package",
            str(package),
            "--port",
            "0",
            "--auth-token",
            token,
            "--max-queued-requests",
            "1",
            "--skip-warmup",
        ]
    if segment:
        command.extend(["--decoder-segment", str(segment)])
    for bundle in bundles:
        command.extend(["--decoder-bundle", str(bundle)])
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        ready = ready_line(process, timeout=300)
        base = ready["url"]
        if segment:
            status, health = request(f"{base}/health", token=token)
            assert status == 200 and health["decoder_segment_loaded"] is True
            assert len(health["decoder_segment_fingerprint"]) == 64
        if bundles:
            status, health = request(f"{base}/health", token=token)
            assert status == 200 and health["decoder_bundle_loaded"] is True
            assert health["decoder_bucket_token_limits"] == sorted(
                health["decoder_bucket_token_limits"]
            )
            assert len(health["decoder_bucket_token_limits"]) == len(bundles)
        request_id = "real-cancellation-smoke"
        body = {
            "model": "wemm-embedding-2b-apple-64",
            "input": "这是用于验证并发、过载和取消行为的长文本。" * 250,
            "request_id": request_id,
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            active = executor.submit(
                request,
                f"{base}/v1/embeddings",
                "POST",
                body,
                token,
                180,
            )
            deadline = time.monotonic() + 5
            running = 0
            while running != 1 and time.monotonic() < deadline:
                time.sleep(0.02)
                health_status, health = request(f"{base}/health", token=token)
                assert health_status == 200
                running = health["running_requests"]
            assert running == 1

            queued = executor.submit(
                request,
                f"{base}/v1/embeddings",
                "POST",
                {"model": "wemm-embedding-2b-apple-64", "input": "queued"},
                token,
                180,
            )
            queued_count = 0
            deadline = time.monotonic() + 5
            while queued_count != 1 and time.monotonic() < deadline:
                time.sleep(0.02)
                health_status, health = request(f"{base}/health", token=token)
                assert health_status == 200
                queued_count = health["queued_requests"]
            assert queued_count == 1

            overloaded, payload = request(
                f"{base}/v1/embeddings",
                method="POST",
                token=token,
                body={"model": "wemm-embedding-2b-apple-64", "input": "queue full"},
            )
            assert overloaded == 429 and payload["error"]["code"] == "service_overloaded"

            cancel_status, _ = request(
                f"{base}/v1/requests/{request_id}",
                method="DELETE",
                token=token,
            )
            assert cancel_status == 202
            cancelled, payload = active.result(timeout=180)
            assert cancelled == 499 and "取消" in payload["error"]["message"]
            assert payload["error"]["code"] == "request_cancelled"
            queued_status, queued_payload = queued.result(timeout=180)
            assert queued_status == 200
            assert len(queued_payload["data"][0]["embedding"]) == 64
        process.terminate()
        assert process.wait(timeout=10) == 0
        print(
            json.dumps(
                {
                    "status": "passed",
                    "real_control_plane": True,
                    "cancel_status": cancel_status,
                    "overload_status": overloaded,
                    "request_status": cancelled,
                    "queued_request_status": queued_status,
                    "graceful_exit": True,
                },
                ensure_ascii=False,
            )
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"native service test failed: {error}", file=sys.stderr)
        raise
