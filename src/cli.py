"""Command-line entry point for the shared CivicStruct inference path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inference import CivicStructInference


def main() -> int:
    parser = argparse.ArgumentParser(description="Structure one civic complaint as JSON.")
    parser.add_argument("complaint", nargs="?", help="complaint text; read stdin when omitted")
    parser.add_argument("--adapter-path", type=Path, help="local registered adapter directory")
    args = parser.parse_args()
    complaint = args.complaint if args.complaint is not None else sys.stdin.read()
    try:
        result = CivicStructInference(args.adapter_path).infer(complaint)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "prediction": None,
                    "raw_response": None,
                    "error": {"type": "load_error", "message": str(exc)},
                    "latency_seconds": 0.0,
                }
            )
        )
        return 1
    print(json.dumps(result.as_dict(), ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
