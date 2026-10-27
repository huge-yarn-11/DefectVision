"""Live webcam inference for DefectVision (NEU-DET steel-surface defects)."""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/best.pt', help='Path to trained YOLO weights')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (usually 0)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (0-1)')
    parser.add_argument('--imgsz', type=int, default=320, help='Inference image size; training used 320')
    parser.add_argument('--device', default='auto', help="'auto', 'cpu', or CUDA device such as '0'")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}\nPut best.pt in the models/ folder or pass --model.")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam. Close other camera apps, allow camera access in Windows settings, or try --camera 1.')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    previous = time.perf_counter()
    smoothed_fps = 0.0
    print('DefectVision live camera is running. Press Q or Esc in the video window to exit.')
    print('Best results require steel-surface samples, stable lighting, and a camera view similar to the training data.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Could not read a frame from the camera.')
                break
            kwargs = dict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            if args.device.lower() != 'auto':
                kwargs['device'] = args.device
            result = model.predict(**kwargs)[0]
            display = result.plot()
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous, 1e-9)
            previous = now
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.85 * smoothed_fps + 0.15 * instant_fps
            count = len(result.boxes) if result.boxes is not None else 0
            status = f'Potential defects: {count} | Processing FPS: {smoothed_fps:.1f}'
            cv2.rectangle(display, (5, 5), (min(display.shape[1] - 5, 540), 42), (25, 25, 25), -1)
            cv2.putText(display, status, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('DefectVision | Live Steel-Surface Inspection', display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
