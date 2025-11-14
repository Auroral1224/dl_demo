#!/usr/bin/env bash
set -euo pipefail

COCO_ROOT="../../../proj1141/coco"

METRICS_FILES=()

declare -a YOLOV5_SIZES=("s" "m" "l" "x")
for SIZE in "${YOLOV5_SIZES[@]}"; do
  MODEL="YOLOv5${SIZE}"
  WEIGHTS="weights/yolov5${SIZE}.pt"
  if [ ! -f "${WEIGHTS}" ]; then
    echo "Skipping ${MODEL}, weights not found at ${WEIGHTS}"
    continue
  fi
  python inference_yolov5.py \
    --coco-root "${COCO_ROOT}" \
    --weights "${WEIGHTS}" \
    --model-name "${MODEL}" \
    --output-json "runs/${MODEL}_metrics.json" \
    --vis-dir "runs/${MODEL}_viz"
  METRICS_FILES+=("runs/${MODEL}_metrics.json")
done

declare -a YOLOV6_SIZES=("n" "s" "m" "l")
for SIZE in "${YOLOV6_SIZES[@]}"; do
  MODEL="YOLOv6${SIZE}"
  WEIGHTS="weights/yolov6${SIZE}.pt"
  if [ ! -f "${WEIGHTS}" ]; then
    echo "Skipping ${MODEL}, weights not found at ${WEIGHTS}"
    continue
  fi
  python inference_yolov6.py \
    --coco-root "${COCO_ROOT}" \
    --weights "${WEIGHTS}" \
    --model-name "${MODEL}" \
    --output-json "runs/${MODEL}_metrics.json" \
    --vis-dir "runs/${MODEL}_viz"
  METRICS_FILES+=("runs/${MODEL}_metrics.json")
done

declare -a YOLOV7_MODELS=("YOLOv7" "YOLOv7x")
declare -a YOLOV7_WEIGHTS=("weights/yolov7.pt" "weights/yolov7x.pt")
for idx in "${!YOLOV7_MODELS[@]}"; do
  MODEL="${YOLOV7_MODELS[$idx]}"
  WEIGHTS="${YOLOV7_WEIGHTS[$idx]}"
  if [ ! -f "${WEIGHTS}" ]; then
    echo "Skipping ${MODEL}, weights not found at ${WEIGHTS}"
    continue
  fi
  python inference_yolov7.py \
    --coco-root "${COCO_ROOT}" \
    --weights "${WEIGHTS}" \
    --model-name "${MODEL}" \
    --output-json "runs/${MODEL}_metrics.json" \
    --vis-dir "runs/${MODEL}_viz"
  METRICS_FILES+=("runs/${MODEL}_metrics.json")
done

if [ "${#METRICS_FILES[@]}" -gt 0 ]; then
  python plot_result.py "${METRICS_FILES[@]}" --output runs/comparison.png
else
  echo "No metrics files generated; skipping plot."
fi
