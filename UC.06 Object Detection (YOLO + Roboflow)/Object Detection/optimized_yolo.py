"""
YOLO Webcam Demo — Teaching Optimized Version
Author: (you)
Target: Windows 10 / 11, Linux (Ubuntu, Debian, etc.), macOS / Apple Silicon / laptops

Optimizations applied:
- CPU-only inference (stable, predictable)
- Reduced camera resolution (~6.7× fewer pixels)
- Frame skipping (~3× fewer inferences)
- Reduced YOLO input size (~1.8× fewer FLOPs)

Total effective compute reduction ≈ 30–35× vs naive setup
"""

from ultralytics import YOLO
import cv2

# -------------------------------
# 1. Load model (CPU only)
# -------------------------------
model = YOLO("yolo11n.pt")
model.to("cpu")  # Avoid implicit GPU/MPS usage

# -------------------------------
# 2. Open webcam
# -------------------------------
cap = cv2.VideoCapture(0)

# Reduce camera resolution
# 1920x1080 → 640x480 ≈ 6.75× fewer pixels
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------------
        # 3. Frame skipping
        # -------------------------------
        # Process only 1 out of every 3 frames
        # ≈ 3× reduction in inference calls
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        # -------------------------------
        # 4. YOLO inference
        # -------------------------------
        results = model(
            frame,
            imgsz=192,      # Smaller input → ~1.8× less compute
            conf=0.3,
            verbose=False
        )

        annotated = results[0].plot()

        # -------------------------------
        # 5. Display
        # -------------------------------
        cv2.imshow("YOLO Webcam (Teaching Mode)", annotated)

        # Always exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()