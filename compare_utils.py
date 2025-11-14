#!/usr/bin/env python3
"""Shared helpers for running YOLO evaluations without repo conflicts."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

COCO80_TO_91 = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


def coco80_to_coco91_class() -> List[int]:
    return list(COCO80_TO_91)


def ensure_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def setup_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(name: str) -> torch.device:
    if name.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "" or name == "0":
        return torch.device("cuda:0")
    return torch.device(name)


def letterbox(
    img: np.ndarray,
    new_shape: int | Tuple[int, int] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


@dataclass
class Timing:
    inference: float
    nms: float


class BaseRunner:
    def __init__(
        self,
        name: str,
        weights: Path,
        device: torch.device,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
    ) -> None:
        ensure_exists(weights, f"{name} weights")
        self.name = name
        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.half = self.device.type != "cpu"
        self.model = None
        self.stride = 32
        self.names: Sequence[str] = []

    def warmup(self) -> None:
        if self.device.type == "cpu":
            return
        dummy = torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device)
        dummy = dummy.half() if self.half else dummy.float()
        with torch.inference_mode():
            self.forward(dummy)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def postprocess(self, predictions: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def predict(self, source: Path | np.ndarray) -> Tuple[torch.Tensor, Timing, np.ndarray]:
        if isinstance(source, (str, Path)):
            frame = cv2.imread(str(source))
            if frame is None:
                raise FileNotFoundError(f"Image not found: {source}")
        else:
            frame = source

        img0 = frame.copy()
        processed, ratio, dwdh = letterbox(img0, self.imgsz)
        img = processed.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        with torch.inference_mode():
            t0 = time.perf_counter()
            preds = self.forward(tensor)
            t1 = time.perf_counter()
            det = self.postprocess(preds)[0]
            t2 = time.perf_counter()
            inf_time = t1 - t0
            nms_time = t2 - t1

        if det is not None and len(det):
            det = det.clone()
            det[:, :4] = scale_boxes(det[:, :4], (ratio, dwdh), img0.shape)
            det = det.cpu()
        else:
            det = torch.zeros((0, 6))

        return det, Timing(inference=inf_time, nms=nms_time), img0


def scale_boxes(boxes: torch.Tensor, ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]], img0_shape: Tuple[int, int, int]) -> torch.Tensor:
    ratio, (dw, dh) = ratio_pad
    gain = ratio[0]
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= gain
    boxes[:, 0].clamp_(0, img0_shape[1])
    boxes[:, 1].clamp_(0, img0_shape[0])
    boxes[:, 2].clamp_(0, img0_shape[1])
    boxes[:, 3].clamp_(0, img0_shape[0])
    return boxes


class CocoSubset:
    def __init__(self, root: Path, split: str, max_images: Optional[int]) -> None:
        self.root = root
        self.split = split
        self.max_images = max_images
        ensure_exists(self.root / "images" / split, f"COCO images for {split}")
        self.ann_path = self.root / "annotations" / f"instances_{split}.json"
        ensure_exists(self.ann_path, "COCO annotation json")
        self.coco = COCO(str(self.ann_path))
        ids = self.coco.getImgIds()
        if max_images is not None:
            ids = ids[:max_images]
        self.samples: List[Tuple[int, Path]] = []
        img_dir = self.root / "images" / split
        for img_id in ids:
            info = self.coco.loadImgs(img_id)[0]
            path = img_dir / info["file_name"]
            if path.exists():
                self.samples.append((img_id, path))
        self.ids = [idx for idx, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[Tuple[int, Path]]:
        return iter(self.samples)

    def take(self, count: int) -> List[Tuple[int, Path]]:
        return self.samples[:count]


def detections_to_coco(det: torch.Tensor, img_id: int, class_mapping: Sequence[int]) -> List[Dict[str, float]]:
    det_np = det.cpu().numpy() if isinstance(det, torch.Tensor) else det
    if det_np.size == 0:
        return []
    boxes = xyxy2xywh(det_np[:, :4])
    boxes[:, :2] -= boxes[:, 2:] / 2.0
    results = []
    for bbox, conf, cls in zip(boxes, det_np[:, 4], det_np[:, 5]):
        cls_idx = int(cls)
        category_id = class_mapping[cls_idx] if cls_idx < len(class_mapping) else cls_idx
        results.append(
            {
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [float(round(x, 3)) for x in bbox],
                "score": float(round(conf, 5)),
            }
        )
    return results


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def evaluate_runner(runner: BaseRunner, dataset: CocoSubset, class_map: Sequence[int]) -> Dict[str, float]:
    preds: List[Dict[str, float]] = []
    timing: List[float] = []
    for img_id, path in tqdm(dataset, desc=f"{runner.name} eval"):
        det, tim, _ = runner.predict(path)
        preds.extend(detections_to_coco(det, img_id, class_map))
        timing.append(tim.inference if tim.inference else 0.0)

    if not preds:
        return {"model": runner.name, "map50": 0.0, "map": 0.0, "time_ms": 0.0}

    coco_dt = dataset.coco.loadRes(preds)
    evaler = COCOeval(dataset.coco, coco_dt, "bbox")
    evaler.params.imgIds = dataset.ids
    evaler.evaluate()
    evaler.accumulate()
    evaler.summarize()

    avg_inf = 1000.0 * float(np.mean(timing))
    return {"model": runner.name, "map50": float(evaler.stats[1]), "map": float(evaler.stats[0]), "time_ms": avg_inf}


def draw_detections(image: np.ndarray, det: torch.Tensor, names: Sequence[str], score_thr: float, max_dets: int) -> np.ndarray:
    canvas = image.copy()
    if det is None or len(det) == 0:
        return canvas
    det_np = det.cpu().numpy()
    keep = det_np[:, 4] >= score_thr
    det_np = det_np[keep][:max_dets]
    palette = plt.get_cmap("tab20")
    for bbox, conf, cls in zip(det_np[:, :4], det_np[:, 4], det_np[:, 5]):
        cls_idx = int(cls)
        color = tuple(int(255 * c) for c in palette((cls_idx % 20) / 20.0)[:3])
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cls_idx] if cls_idx < len(names) else cls_idx}:{conf:.2f}"
        cv2.putText(canvas, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return canvas


def visualize_images(
    runner: BaseRunner,
    dataset: CocoSubset,
    out_dir: Path,
    score_thr: float,
    max_dets: int,
    count: int,
) -> None:
    samples = dataset.take(count)
    if not samples:
        return
    for _, path in samples:
        det, _, img0 = runner.predict(path)
        annotated = draw_detections(img0, det, runner.names, score_thr, max_dets)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{runner.name} - {path.name}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{runner.name}_{Path(path).stem}.jpg", dpi=200)
        plt.close(fig)


def visualize_video(
    runner: BaseRunner,
    video_path: Path,
    out_path: Path,
    score_thr: float,
    max_dets: int,
    max_frames: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames = 0
    pbar = tqdm(total=max_frames, desc=f"{runner.name} video")
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        det, _, _ = runner.predict(frame)
        annotated = draw_detections(frame, det, runner.names, score_thr, max_dets)
        writer.write(annotated)
        frames += 1
        pbar.update(1)
    pbar.close()
    writer.release()
    cap.release()


def load_names_from_yaml(path: Path) -> Sequence[str]:
    if not path.exists():
        return [str(i) for i in range(80)]
    data = yaml.safe_load(path.read_text())
    names = data.get("names")
    if isinstance(names, (list, tuple)):
        return list(names)
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return [str(i) for i in range(80)]
