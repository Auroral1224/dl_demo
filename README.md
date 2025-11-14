# YOLO Model Benchmark Toolkit

```
.
├── env.ipynb                 # environment setup + weight download cell
├── run.sh                    # batch evaluation over all YOLO variants
├── inference_yolov5.py       # YOLOv5 COCO val helper
├── inference_yolov6.py       # YOLOv6 COCO val helper
├── inference_yolov7.py       # YOLOv7 COCO val helper
├── plot_result.py            # aggregates metrics JSONs into comparison plot
├── video_detect_yolov7.py    # YOLOv7 video demo
├── compare_utils.py          # shared dataset/inference utilities
├── weights/                  # downloaded checkpoints (ignored by git)
├── runs/                     # outputs: metrics, plots, visualizations
├── YOLOv6/, yolov5/, yolov7/ # upstream repos with original code
└── t.mp4                     # sample video for video_detect_yolov7.py
```

This repo contains a lightweight harness for evaluating pretrained YOLOv5/YOLOv6/YOLOv7 models on COCO val2017 plus a simple video demo runner.

## 1. Environment
Follow `env.ipynb` to create the `DL_YOLO` conda env and install requirements. The notebook already documents the CUDA check, repo cloning, etc.

## 2. Download weights
At the bottom of `env.ipynb` you’ll find a cell that downloads all required checkpoints into `weights/`:

```
%%bash
mkdir -p weights
# YOLOv5
wget -O weights/yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
...
# YOLOv7
wget -O weights/yolov7x.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
```

Run that cell once. The benchmark scripts assume every file lives under the shared `weights/` directory.

## 3. Run the full benchmark

```
bash run.sh  # ensure COCO path/weights paths inside match your setup
```

This will:
1. Loop through YOLOv5 {s,m,l,x}, YOLOv6 {n,s,m,l}, and YOLOv7 {regular,x}
2. Evaluate each model on COCO val2017 using `inference_yolo*.py` (weights loaded from `weights/…`)
3. Save metrics JSONs under `runs/<model>_metrics.json` and sample visualizations per model under `runs/<model>_viz/`
4. Aggregate all JSONs via `plot_result.py` into `runs/comparison.png`, plotting mAP vs. inference time for each family.

If a weight file is missing, `run.sh` skips that model with a warning.

## 4. Video detection demo

```
python video_detect_yolov7.py --video t.mp4 --output runs/video/yolov7.mp4
```

Customize `--weights`, `--imgsz`, `--conf`, `--max-frames`, or `--device` as needed. The script reuses the YOLOv7 runner to annotate every frame of the input clip and saves the result under `runs/video/`.

## 5. Custom runs
Each inference script accepts `--weights`, `--model-name`, `--max-images`, etc. Example:

```
python inference_yolov5.py \
  --coco-root /path/to/coco \
  --weights weights/yolov5m.pt \
  --model-name YOLOv5m_custom \
  --max-images 128
```

This produces metrics at `runs/YOLOv5m_custom_metrics.json` and visuals under `runs/YOLOv5m_custom_viz/`.

## Outputs
* Metrics per model: `runs/<model>_metrics.json`
* Visualizations: `runs/<model>_viz/*.jpg`
* Trade-off plot: `runs/comparison.png`
* Video demo: `runs/video/*.mp4`

