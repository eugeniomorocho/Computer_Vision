import time
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -------------------------------
# 1. Load YOLO model (raw network only)
# -------------------------------
yolo = YOLO("yolo11n.pt")
net = yolo.model  # <-- this is the pure CNN (no NMS)

# -------------------------------
# 2. Create synthetic input
# -------------------------------
# Simulates a 256x256 RGB image
x_cpu = torch.randn(1, 3, 256, 256)

devices = {
    "CPU": "cpu",
    "MPS (Apple GPU)": "mps" if torch.backends.mps.is_available() else None
}

fps = {}

for name, device in devices.items():
    if device is None:
        continue

    net = net.to(device)
    x = x_cpu.to(device)

    # -------------------------------
    # 3. Warmup
    # -------------------------------
    for _ in range(10):
        _ = net(x)

    # -------------------------------
    # 4. Timed runs
    # -------------------------------
    runs = 50
    start = time.time()

    for _ in range(runs):
        _ = net(x)

    elapsed = time.time() - start
    fps[name] = runs / elapsed

# -------------------------------
# 5. Plot
# -------------------------------
plt.bar(fps.keys(), fps.values())
plt.ylabel("FPS (CNN forward only)")
plt.title("YOLO Forward Pass: CPU vs Apple GPU (MPS)")
plt.show()
