#!/usr/bin/env python3
"""Evaluate YOLOv6 on COCO val."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import torch

from compare_utils import (
    BaseRunner,
    CocoSubset,
    coco80_to_coco91_class,
    evaluate_runner,
    load_names_from_yaml,
    resolve_device,
    setup_output_dir,
    visualize_images,
)

ROOT = Path(__file__).resolve().parent
YOLOV6_DIR = ROOT / "YOLOv6"
sys.path.append(str(YOLOV6_DIR))

from yolov6.layers.common import DetectBackend, RepVGGBlock  # type: ignore
from yolov6.utils.nms import non_max_suppression  # type: ignore


class YoloV6Runner(BaseRunner):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        backend = DetectBackend(str(self.weights), device=self.device)
        model = backend.model
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, "recompute_scale_factor"):
                layer.recompute_scale_factor = None
        if self.half:
            model.half()
        self.backend = backend
        self.model = model
        self.names = self._resolve_names(model)
        self.stride = int(model.stride.max())
        self.warmup()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.backend(tensor)

    def postprocess(self, predictions: torch.Tensor) -> List[torch.Tensor]:
        return non_max_suppression(
            predictions,
            conf_thres=self.conf,
            iou_thres=self.iou,
            multi_label=True,
            max_det=self.max_det,
        )

    def _resolve_names(self, model) -> Sequence[str]:
        if hasattr(model, "names"):
            return model.names
        return load_names_from_yaml(YOLOV6_DIR / "data" / "coco.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv6 COCO evaluation helper.")
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolov6s.pt")
    parser.add_argument("--model-name", type=str, default="YOLOv6s")
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
    runner = YoloV6Runner(
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
