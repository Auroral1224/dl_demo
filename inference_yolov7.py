#!/usr/bin/env python3
"""Evaluate YOLOv7 on COCO val."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch

from compare_utils import (
    BaseRunner,
    CocoSubset,
    coco80_to_coco91_class,
    evaluate_runner,
    resolve_device,
    setup_output_dir,
    visualize_images,
)

ROOT = Path(__file__).resolve().parent
YOLOV7_DIR = ROOT / "yolov7"
sys.path.append(str(YOLOV7_DIR))

from yolov7.models.experimental import attempt_load  # type: ignore
from yolov7.utils.general import non_max_suppression  # type: ignore


class YoloV7Runner(BaseRunner):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        model = attempt_load(str(self.weights), map_location=self.device)
        model.eval()
        if self.half:
            model.half()
        self.model = model
        self.names = model.names if hasattr(model, "names") else list(range(model.nc))
        self.stride = int(model.stride.max())
        self.warmup()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        preds, _ = self.model(tensor)
        return preds

    def postprocess(self, predictions: torch.Tensor) -> List[torch.Tensor]:
        return non_max_suppression(predictions, conf_thres=self.conf, iou_thres=self.iou)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv7 COCO evaluation helper.")
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolov7.pt")
    parser.add_argument("--model-name", type=str, default="YOLOv7")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--vis-dir", type=Path)
    parser.add_argument("--vis-count", type=int, default=2)
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.vis_dir is None:
        args.vis_dir = ROOT / "runs" / f"{args.model_name}_viz"
    if args.output_json is None:
        args.output_json = ROOT / "runs" / f"{args.model_name}_metrics.json"
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    max_imgs = None if args.max_images in (-1, None) else args.max_images

    dataset = CocoSubset(args.coco_root, "val2017", max_imgs)
    class_mapping = coco80_to_coco91_class()
    runner = YoloV7Runner(
        name=args.model_name,
        weights=args.weights,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )
    metrics = evaluate_runner(runner, dataset, class_mapping)
    print(f"{runner.name}: mAP@0.5:0.95={metrics['map']:.3f}, mAP@0.5={metrics['map50']:.3f}, avg inf={metrics['time_ms']:.2f}ms")

    setup_output_dir(args.output_json.parent)
    args.output_json.write_text(json.dumps(metrics, indent=2))

    if args.vis_count > 0:
        setup_output_dir(args.vis_dir)
        visualize_images(runner, dataset, args.vis_dir, args.score_thr, args.max_det, args.vis_count)

if __name__ == "__main__":
    main()
