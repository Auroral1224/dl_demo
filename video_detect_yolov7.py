#!/usr/bin/env python3
"""Run YOLOv7 video detection on a single clip."""
from __future__ import annotations

import argparse
from pathlib import Path

from compare_utils import resolve_device, setup_output_dir, visualize_video
from inference_yolov7 import YoloV7Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv7 video detection helper.")
    parser.add_argument("--video", type=Path, default=Path("t.mp4"), help="Input video path.")
    parser.add_argument("--weights", type=Path, default=Path("weights") / "yolov7.pt")
    parser.add_argument("--output", type=Path, default=Path("runs") / "video" / "yolov7_detect.mp4")
    parser.add_argument("--model-name", type=str, default="YOLOv7-video")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=10**9, help="Limit processed frames (default processes all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    runner = YoloV7Runner(
        name=args.model_name,
        weights=args.weights,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )
    setup_output_dir(args.output.parent)
    visualize_video(runner, args.video, args.output, args.score_thr, args.max_det, args.max_frames)


if __name__ == "__main__":
    main()
