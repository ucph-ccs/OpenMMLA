#!/usr/bin/env python3
"""Smoke-test the frame analyzer's features endpoint (POST /vllm/features).

Sends one or more frames (simultaneous captures from different camera angles) and
prints, per frame, the persons found: the AprilTag each wears, their head yaw and where
their gaze lands; no VLM is involved. The full answer (skeletons included) can be saved.

Example:
    python frame_features.py front.jpg side.jpg --angles front,side \
      --zones '{"table": [[0, 0.55], [1, 0.55], [1, 1], [0, 1]]}' --out features.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send frames to the frame analyzer's features endpoint.")
    parser.add_argument("images", nargs="+", help="frame image path(s), one per camera angle")
    parser.add_argument("--server-url", default="http://localhost:5007/vllm",
                        help="analyzer endpoint; /features is appended (default: http://localhost:5007/vllm)")
    parser.add_argument("--session-id", default="vfa_example", help="session id reported to the analyzer")
    parser.add_argument("--angles", default=None,
                        help="comma-separated angle names, one per image (default: angle_1, angle_2, ...)")
    parser.add_argument("--zones", default=None,
                        help='JSON of named polygons a gaze may land in, {"table": [[x, y], ...]} for every '
                             'frame or {"front": {"table": [...]}} per angle, in pixels or in [0, 1]')
    parser.add_argument("--inout-threshold", type=float, default=None,
                        help="below it a gaze counts as out of frame (default: the server's)")
    parser.add_argument("--no-keypoints", action="store_true", help="leave the skeletons out of the answer")
    parser.add_argument("--out", default=None, help="write the full answer as JSON to this file")
    parser.add_argument("--render", default=None,
                        help="a folder to draw the answer into: each frame with its boxes, tags, skeletons, head "
                             "yaws and gaze lines, as <frame>_features.jpg (needs opencv)")
    return parser


def render(image_path: str, frame: dict, out_dir: str) -> str:
    """draw one frame's answer onto its image (openmmla.utils.video.overlay) and save it."""
    import cv2
    from openmmla.utils.video.overlay import draw_features
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"not an image: {image_path}")
    draw_features(image, frame)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + "_features.jpg")
    cv2.imwrite(out_path, image)
    return out_path


def main() -> int:
    args = get_parser().parse_args()

    for path in args.images:
        if not os.path.isfile(path):
            print(f"Image not found: {path}", file=sys.stderr)
            return 1

    angles = ([item.strip() for item in args.angles.split(",")] if args.angles
              else [f"angle_{i + 1}" for i in range(len(args.images))])
    if len(angles) != len(args.images):
        print("Number of angles must match number of images.", file=sys.stderr)
        return 1

    data = {"session_id": args.session_id, "angles": json.dumps(angles)}
    if args.zones:
        try:
            data["zones"] = json.dumps(json.loads(args.zones))
        except json.JSONDecodeError as exc:
            print(f"Invalid --zones JSON: {exc}", file=sys.stderr)
            return 1
    if args.inout_threshold is not None:
        data["inout_threshold"] = str(args.inout_threshold)
    if args.no_keypoints:
        data["keypoints"] = "false"

    files = [("images", (os.path.basename(path), open(path, "rb"), "image/jpeg")) for path in args.images]
    url = args.server_url.rstrip("/") + "/features"
    try:
        response = requests.post(url, files=files, data=data, timeout=120)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for _, (_, handle, _) in files:
            handle.close()

    if response.status_code != 200:
        print(f"{url} answered {response.status_code}: {response.text[:300]}", file=sys.stderr)
        return 1
    answer = response.json()

    print(f"pose model: {answer.get('pose_model')}   gaze model: {'on' if answer.get('gaze') else 'off'}")
    for frame in answer.get("frames", []):
        print(f"\n[{frame['angle']}] {frame['width']}x{frame['height']}  tags: {frame['tags']}")
        if frame.get("gaze_error"):
            print(f"  gaze model failed on this frame: {frame['gaze_error']}")
        for person in frame["persons"]:
            gaze = person["gaze"]
            target = gaze["target"]
            where = target["category"] + (f" ({target['person_id']})" if target.get("person_id") else "") \
                + (f" ({target['zone']})" if target.get("zone") else "")
            keypoints = person.get("keypoints")
            seen = f"{sum(1 for _, _, confidence in keypoints if confidence >= 0.3)}/17" if keypoints else "left out"
            inout = gaze.get("inout")
            inout_text = f"  in {inout:.2f}" if inout is not None else ""
            print(f"  person {person['person_id']:>10}  tag {person['tag_id']}  box {[int(v) for v in person['bbox']]}"
                  f"  keypoints {seen}  head_yaw {person['head_yaw']}  gaze -> {where}{inout_text}")
        for pair, distances in frame["pairs"].items():
            print(f"  pair {pair}: gaze_distance {distances['gaze_distance']}  hand_distance {distances['hand_distance']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(answer, handle, indent=1)
        print(f"\nwrote {args.out}")
    if args.render:
        for image_path, frame in zip(args.images, answer.get("frames", [])):
            print(f"drew {render(image_path, frame, args.render)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
