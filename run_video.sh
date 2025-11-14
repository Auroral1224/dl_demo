#!/usr/bin/env bash
python video_detect_yolov7.py \
  --video t.mp4 \
  --max-frames 600 \
  --imgsz 512 \
  --conf 0.35 \
  --device 0