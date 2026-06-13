#!/usr/bin/env python3
"""Smoke-test the multi-angle VLLM frame analyzer service.

Sends one or more frames (e.g. simultaneous captures from different camera
angles) to the analyzer endpoint and prints the observations, classifications,
and justifications returned by the VLM pipeline.

Example:
    python analyze_video_frame.py front.jpg side.jpg \
      --angles front,side \
      --angle-descriptions "Front view of the table,Side view from the door" \
      --participant-descriptions '{"1": "person with red shirt", "2": "person with glasses"}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send multi-angle frames to the VLLM frame analyzer service."
    )
    parser.add_argument("images", nargs="+", help="frame image path(s), one per camera angle")
    parser.add_argument("--server-url", default="http://localhost:5007/vllm",
                        help="analyzer endpoint (default: http://localhost:5007/vllm)")
    parser.add_argument("--session-id", default="vfa_example",
                        help="session id reported to the analyzer")
    parser.add_argument("--angles", default=None,
                        help="comma-separated angle names, one per image (default: angle_1, angle_2, ...)")
    parser.add_argument("--angle-descriptions", default=None,
                        help="comma-separated angle descriptions, one per image")
    parser.add_argument("--participant-descriptions", default="{}",
                        help='JSON dict mapping participant ids to descriptions, e.g. {"1": "red shirt"}')
    return parser


def main() -> int:
    args = get_parser().parse_args()

    for path in args.images:
        if not os.path.isfile(path):
            print(f"Image not found: {path}", file=sys.stderr)
            return 1

    angles = (
        [item.strip() for item in args.angles.split(",")]
        if args.angles else [f"angle_{i + 1}" for i in range(len(args.images))]
    )
    angle_descriptions = (
        [item.strip() for item in args.angle_descriptions.split(",")]
        if args.angle_descriptions else [f"Image from {angle} perspective" for angle in angles]
    )
    if len(angles) != len(args.images) or len(angle_descriptions) != len(args.images):
        print("Number of angles/descriptions must match number of images.", file=sys.stderr)
        return 1

    try:
        participant_descriptions = json.loads(args.participant_descriptions)
    except json.JSONDecodeError as exc:
        print(f"Invalid --participant-descriptions JSON: {exc}", file=sys.stderr)
        return 1

    files = [
        ("images", (os.path.basename(path), open(path, "rb"), "image/jpeg"))
        for path in args.images
    ]
    data = {
        "session_id": args.session_id,
        "angles": json.dumps(angles),
        "angle_descriptions": json.dumps(angle_descriptions),
        "participant_descriptions": json.dumps(participant_descriptions),
    }

    try:
        response = requests.post(args.server_url, files=files, data=data, timeout=300)
    finally:
        for _, (_, handle, _) in files:
            handle.close()

    if response.status_code != 200:
        print(f"Request failed ({response.status_code}): {response.text}", file=sys.stderr)
        return 1

    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
