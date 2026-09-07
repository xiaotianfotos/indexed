#!/usr/bin/env python3
"""Benchmark the packaged Swift helper as an external Electron-style process."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import platform
import select
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-runs", type=int, default=2)
    parser.add_argument("--text-runs", type=int, default=5)
    parser.add_argument("--image-runs", type=int, default=5)
    parser.add_argument("--mixed-runs", type=int, default=3)
    parser.add_argument("--inter-process-delay", type=float, default=1.0)
    return parser.parse_args()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cache_snapshot(path: Path) -> dict[str, int | list[str]]:
    entries = sorted(item.name for item in path.iterdir()) if path.exists() else []
    return {"entries": entries, "bytes": directory_bytes(path) if path.exists() else 0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def request(url: str, token: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    value = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(value, timeout=180) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error


def ready_line(process: subprocess.Popen[str], timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    assert process.stdout is not None
    while time.monotonic() < deadline:
        readable, _, _ = select.select([process.stdout], [], [], 0.2)
        if readable:
            line = process.stdout.readline()
            if line:
                return json.loads(line)
        if process.poll() is not None:
            raise RuntimeError(f"helper exited before ready with {process.returncode}")
    raise TimeoutError("helper did not become ready")


class RSSSampler:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self.values: list[int] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        while not self.stop_event.wait(0.05):
            if self.process.poll() is not None:
                return
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(self.process.pid)],
                capture_output=True,
                text=True,
                check=False,
            )
            value = result.stdout.strip()
            if value:
                self.values.append(int(value) * 1024)


def median_timings(rows: list[dict]) -> dict[str, float]:
    keys = rows[0]["indexed"]["timings_ms"].keys()
    return {
        key: statistics.median(row["indexed"]["timings_ms"][key] for row in rows)
        for key in keys
    }


def run_process(
    *,
    binary: Path,
    package: Path,
    segment: Path,
    cache: Path,
    image_data_uri: str,
    text_runs: int,
    image_runs: int,
    mixed_runs: int,
) -> dict:
    token = "native-benchmark-local-token"
    command = [
        str(binary),
        "serve",
        "--package",
        str(package),
        "--port",
        "0",
        "--default-dimension",
        "2048",
        "--decoder-segment",
        str(segment),
        "--decoder-minimum-tokens",
        "224",
        "--coreml-cache",
        str(cache),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=binary.parent,
        env={
            **__import__("os").environ,
            "INDEXED_APPLE_EMBEDDING_AUTH_TOKEN": token,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr: list[str] = []
    assert process.stderr is not None
    drain = threading.Thread(
        target=lambda: stderr.extend(line.rstrip() for line in process.stderr), daemon=True
    )
    drain.start()
    sampler = RSSSampler(process)
    sampler.start()
    try:
        ready = ready_line(process)
        ready_wall_seconds = time.perf_counter() - started
        base = ready["url"]
        health = request(f"{base}/health", token)
        text_body = {
            "model": "wemm-embedding-2b-apple-2048",
            "input": "一只坐在窗边的猫",
        }
        image_body = {
            "model": "wemm-embedding-2b-apple-2048",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": "Represent this image."},
                    ],
                }
            ],
        }
        mixed_body = {
            "model": "wemm-embedding-2b-apple-2048",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {
                            "type": "text",
                            "text": (
                                "Use the reference only for the mug shape and composition. "
                                "Retrieve a matte blue mug on a white marble counter beside "
                                "a closed dark green hardcover book."
                            ),
                        },
                    ],
                }
            ],
        }
        text_rows = [request(f"{base}/v1/embeddings", token, text_body) for _ in range(text_runs)]
        image_rows = [
            request(f"{base}/v1/embeddings", token, image_body) for _ in range(image_runs)
        ]
        mixed_rows = [
            request(f"{base}/v1/embeddings", token, mixed_body) for _ in range(mixed_runs)
        ]
        all_rows = text_rows + image_rows + mixed_rows
        assert all(len(row["data"][0]["embedding"]) == 2048 for row in all_rows)
        assert all(row["indexed"]["timings_ms"]["coreml_decoder"] > 0 for row in mixed_rows)
        process.terminate()
        graceful = process.wait(timeout=15) == 0
        return {
            "ready": ready,
            "health": health,
            "ready_wall_seconds": ready_wall_seconds,
            "rss_peak_bytes": max(sampler.values, default=0),
            "text_median_ms": median_timings(text_rows),
            "image_median_ms": median_timings(image_rows),
            "mixed_median_ms": median_timings(mixed_rows),
            "text_first_ms": text_rows[0]["indexed"]["timings_ms"],
            "image_first_ms": image_rows[0]["indexed"]["timings_ms"],
            "mixed_first_ms": mixed_rows[0]["indexed"]["timings_ms"],
            "prompt_tokens": {
                "text": text_rows[0]["usage"]["prompt_tokens"],
                "image": image_rows[0]["usage"]["prompt_tokens"],
                "mixed": mixed_rows[0]["usage"]["prompt_tokens"],
            },
            "graceful_exit": graceful,
            "stderr": stderr,
        }
    finally:
        sampler.stop()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        drain.join(timeout=2)


def sysctl(name: str) -> str:
    return subprocess.check_output(["sysctl", "-n", name], text=True).strip()


def main() -> None:
    args = parse_args()
    if min(args.process_runs, args.text_runs, args.image_runs, args.mixed_runs) < 1:
        raise ValueError("run counts must be positive")
    binary = args.binary.resolve()
    package = args.package.resolve()
    segment = args.segment.resolve()
    image = args.image.resolve()
    with tempfile.TemporaryDirectory(prefix="indexed-coreml-cache-") as cache_value:
        cache = Path(cache_value)
        runs = []
        for index in range(args.process_runs):
            run = run_process(
                binary=binary,
                package=package,
                segment=segment,
                cache=cache,
                image_data_uri=image_uri(image),
                text_runs=args.text_runs,
                image_runs=args.image_runs,
                mixed_runs=args.mixed_runs,
            )
            run["coreml_cache_after_run"] = cache_snapshot(cache)
            runs.append(run)
            if index + 1 < args.process_runs:
                time.sleep(args.inter_process_delay)
    report = {
        "schema_version": 1,
        "machine": {
            "chip": sysctl("machdep.cpu.brand_string"),
            "physical_memory_bytes": int(sysctl("hw.memsize")),
            "macos": platform.mac_ver()[0],
        },
        "artifacts": {
            "binary": str(binary),
            "binary_sha256": sha256(binary),
            "package": str(package),
            "package_bytes": directory_bytes(package),
            "segment": str(segment),
            "segment_bytes": directory_bytes(segment),
        },
        "configuration": {
            "dimension": 2048,
            "decoder_minimum_tokens": 224,
            "process_runs": args.process_runs,
            "text_runs_per_process": args.text_runs,
            "image_runs_per_process": args.image_runs,
            "mixed_runs_per_process": args.mixed_runs,
            "inter_process_delay_seconds": args.inter_process_delay,
            "first_process_uses_empty_coreml_cache": True,
            "later_processes_reuse_same_coreml_cache": True,
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
