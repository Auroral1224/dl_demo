#!/usr/bin/env python3
"""Plot comparison chart from individual YOLO inference JSON results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

FAMILIES = {
    "YOLOv5": ["YOLOv5s", "YOLOv5m", "YOLOv5l", "YOLOv5x"],
    "YOLOv6": ["YOLOv6n", "YOLOv6s", "YOLOv6m", "YOLOv6l"],
    "YOLOv7": ["YOLOv7", "YOLOv7x"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot inference trade-offs from JSON metrics.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Paths to JSON metrics files.")
    parser.add_argument("--output", type=Path, default=Path("runs") / "comparison.png")
    return parser.parse_args()


def load_metrics(paths: List[Path]) -> List[Dict[str, float]]:
    items = []
    for path in paths:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            model = data.get("model") or path.stem
            items.append(
                {
                    "model": model,
                    "map": float(data.get("map", 0.0)),
                    "map50": float(data.get("map50", 0.0)),
                    "time_ms": float(data.get("time_ms", 0.0)),
                }
            )
    return items


def plot(metrics: List[Dict[str, float]], output: Path) -> None:
    if not metrics:
        print("No metrics to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(FAMILIES)))

    used_models = set()
    for color, (family, order) in zip(colors, FAMILIES.items()):
        points = []
        for name in order:
            match = next((m for m in metrics if m["model"] == name), None)
            if match:
                points.append(match)
                used_models.add(name)
        if not points:
            continue
        xs = [p["time_ms"] for p in points]
        ys = [p["map"] for p in points]
        ax.plot(xs, ys, marker="o", label=family, color=color)
        for p in points:
            ax.text(p["time_ms"], p["map"], p["model"])

    for item in metrics:
        if item["model"] in used_models:
            continue
        ax.scatter(item["time_ms"], item["map"], color="gray", label=item["model"])
        ax.text(item["time_ms"], item["map"], item["model"])

    ax.set_xlabel("Avg inference time (ms/img)")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output}")


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.inputs)
    for item in metrics:
        print(f"{item['model']}: mAP@0.5:0.95={item['map']:.3f}, mAP@0.5={item['map50']:.3f}, time={item['time_ms']:.2f}ms")
    plot(metrics, args.output)


if __name__ == "__main__":
    main()
