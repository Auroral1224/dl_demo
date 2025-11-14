#!/usr/bin/env python3
"""Evaluate YOLOv5 on COCO val without conflicting with other YOLO repos."""
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
YOLOV5_DIR = ROOT / "yolov5"
sys.path.append(str(YOLOV5_DIR))

from yolov5.models.common import DetectMultiBackend  # type: ignore
from yolov5.utils.general import check_img_size, non_max_suppression  # type: ignore
from yolov5.utils.torch_utils import select_device  # type: ignore


class YoloV5Runner(BaseRunner):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        select_device(str(self.device))
        model = DetectMultiBackend(
            str(self.weights),
            device=self.device,
            dnn=False,
            data=str(YOLOV5_DIR / "data" / "coco.yaml"),
            fp16=self.half,
        )
        self.model = model
        self.names = model.names
        self.stride = int(model.stride)
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.warmup()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)

    def postprocess(self, predictions: torch.Tensor) -> List[torch.Tensor]:
        return non_max_suppression(
            predictions,
            conf_thres=self.conf,
            iou_thres=self.iou,
            max_det=self.max_det,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv5 COCO evaluation helper.")
    parser.add_argument("--coco-root", type=Path, required=True, help="COCO root containing images/ and annotations/.")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolov5s.pt")
    parser.add_argument("--model-name", type=str, default="YOLOv5s")
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
    runner = YoloV5Runner(
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
