# DefectVision — Steel-Surface Defect Detection

A student project using YOLOv8n and the NEU-DET steel-surface dataset to detect six categories of surface defects. It includes the Kaggle training notebook, trained weights, example results, image inference, and a laptop-webcam demo.

> **Scope:** this is an educational prototype. It has not been validated for factory quality control, safety decisions, or automatic rejection systems. Performance on a laptop webcam may differ from validation results because lighting, scale, focus, and surface appearance change.

## Defect classes

- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

## Repository layout

```text
DefectVision-GitHub-Ready/
├── models/best.pt
├── notebooks/NEU_DET_YOLOv8_Training.ipynb
├── results/
├── webcam_detect.py
├── predict_image.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Setup on Windows

Use Python 3.10–3.12 if possible for a smoother package installation experience. From this folder in Command Prompt or the VS Code terminal:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` is not recognized, install Python from https://www.python.org/downloads/ and enable **Add Python to PATH** during setup.

## 2. Run a prediction on one image

```bat
python predict_image.py "path\to\steel_sample.jpg"
```

The annotated result is saved to `results/prediction.jpg`. Replace the sample path with a real image file on your computer.

## 3. Run live laptop-webcam detection

```bat
python webcam_detect.py
```

Press **Q** or **Esc** while the video window is focused to exit. Options:

```bat
python webcam_detect.py --camera 1
python webcam_detect.py --conf 0.35 --imgsz 320
python webcam_detect.py --device cpu
```

The default camera index is `0`. If it does not open, close other camera apps, allow desktop apps to use the camera in Windows privacy settings, and try another camera index.

### Important live-demo notes

- The model was trained on small, grayscale NEU-DET images of steel surfaces. Point the camera at a steel sample; arbitrary objects are out of distribution.
- Use steady lighting, close focus, and a similar view to the training images.
- The confidence threshold is configurable and should be validated on separate images from the target camera.
- `Processing FPS` is an approximate inference/display rate, not an industrial line-speed guarantee.
- This repository does not include PLC controls, industrial-camera synchronization, production monitoring, or a validated defect-rejection mechanism.

## 4. Reproduce training in Kaggle

1. Open the notebook at `notebooks/NEU_DET_YOLOv8_Training.ipynb` in Kaggle or upload it as a new Kaggle notebook.
2. Enable Internet if permitted by the notebook environment, and enable a GPU accelerator if available.
3. Run cells in order. The notebook downloads NEU-DET, converts Pascal VOC XML bounding boxes to YOLO labels, trains YOLOv8n, validates it, and exports `best.pt`.
4. Download the resulting weights and replace `models/best.pt` only after checking the new model's validation results.

Kaggle download links, runtime availability, and dataset hosting can change. The notebook currently downloads the dataset using a Kaggle dataset URL; if that cell fails, attach the NEU-DET dataset through Kaggle's **Add Input** panel and update the dataset paths in the notebook.

## Reported evaluation

The original README reported mAP50 `0.7710` and mAP50-95 `0.445`. These values are inherited from the submitted project and have not been independently reproduced here. Report results from a held-out test set for any formal report.

## GitHub publishing checklist

- Commit the notebook, inference scripts, README, requirements, results plots, and model weights.
- Do not upload the full dataset, virtual environment, Kaggle credentials, or API tokens.
- Verify dataset terms and cite the NEU-DET source in your academic report.
- The weights file is currently small enough for a normal GitHub repository. GitHub blocks individual files larger than 100 MB.
