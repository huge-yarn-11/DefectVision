"""Run DefectVision on one image and save the annotated prediction."""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('image', help='Path to an image of a steel surface')
parser.add_argument('--model', default='models/best.pt', help='Path to model weights')
parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
parser.add_argument('--imgsz', type=int, default=320, help='Inference image size')
parser.add_argument('--output', default='results/prediction.jpg', help='Output image path')
args = parser.parse_args()
root = Path(__file__).resolve().parent
model_path = Path(args.model)
if not model_path.is_absolute(): model_path = root / model_path
image_path = Path(args.image)
if not image_path.is_absolute(): image_path = Path.cwd() / image_path
if not model_path.is_file(): raise FileNotFoundError(f'Model not found: {model_path}')
if not image_path.is_file(): raise FileNotFoundError(f'Image not found: {image_path}')
model = YOLO(str(model_path))
result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
annotated = result.plot()
out = Path(args.output)
if not out.is_absolute(): out = root / out
out.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(out), annotated): raise RuntimeError(f'Could not save result to {out}')
print(f'Detections: {len(result.boxes) if result.boxes is not None else 0}')
print(f'Saved annotated image: {out}')
