#!/usr/bin/env python3
"""Remove workstation-specific paths and private hosts from public result JSON."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


PRIVATE_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?![\d.])")
USER_HOME = re.compile(r"/Users/[^/]+")
VOLUME_ROOT = re.compile(r"/Volumes/[^/]+")
SENSITIVE_KEYS = {"api_key", "apikey", "authorization", "token", "access_token"}


def sanitize_string(value: str, replacements: list[tuple[str, str]]) -> str:
    for source, replacement in replacements:
        if source:
            value = value.replace(source, replacement)
    value = USER_HOME.sub("<USER_HOME>", value)
    value = VOLUME_ROOT.sub("<VOLUME>", value)

    def replace_private_host(match: re.Match[str]) -> str:
        raw = match.group(0)
        host, separator, port = raw.partition(":")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return raw
        if address.is_loopback:
            return raw
        if address.is_private:
            return f"<PRIVATE_HOST>{separator}{port}" if separator else "<PRIVATE_HOST>"
        return raw

    return PRIVATE_IPV4.sub(replace_private_host, value)


def sanitize(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        return sanitize_string(value, replacements)
    if isinstance(value, list):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = re.sub(r"(?<!^)([A-Z])", r"_\1", key).lower().replace("-", "_")
            secret_shaped = normalized_key in SENSITIVE_KEYS or normalized_key.endswith(
                ("_api_key", "_access_token", "_auth_token")
            )
            if secret_shaped and item:
                raise ValueError(f"refusing to publish populated sensitive field: {key}")
            result[key] = sanitize(item, replacements)
        return result
    return value


def write_json(path: Path, value: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexed-root", type=Path)
    parser.add_argument("--wemm-workspace", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    component = Path(__file__).resolve().parent
    results = component / "results"
    indexed_root = (args.indexed_root or component.parent.parent).expanduser().resolve()
    replacements = [(str(indexed_root), "<INDEXED_ROOT>")]
    if args.wemm_workspace:
        replacements.append((str(args.wemm_workspace.expanduser().resolve()), "<WEMM_WORKSPACE>"))

    changed: list[str] = []
    for path in sorted(results.glob("*.json")):
        original = json.loads(path.read_text(encoding="utf-8"))
        public = sanitize(original, replacements)
        encoded = json.dumps(public, ensure_ascii=False, indent=2) + "\n"
        if encoded == path.read_text(encoding="utf-8"):
            continue
        changed.append(path.name)
        if not args.check:
            write_json(path, public)

    if args.check and changed:
        raise SystemExit("unsanitized result JSON: " + ", ".join(changed))
    print(json.dumps({"status": "clean" if not changed else "sanitized", "files": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
