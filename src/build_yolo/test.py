import cv2
import torch
import torch.nn as nn
import numpy as np
from src.build_yolo.model import ResNetFistDetector  # đường dẫn model bạn lưu

# -------------------------
# Load model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetFistDetector().to(device)
model.load_state_dict(torch.load("src/build_yolo/tinyyolo_from_scratch.pth", map_location=device))
model.eval()

# -------------------------
# Helper function: preprocess frame
# -------------------------
def preprocess(frame, input_size=224):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    img = img.unsqueeze(0)  # batch_size=1
    return img.to(device)

# -------------------------
# Webcam inference
# -------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Preprocess
    input_tensor = preprocess(frame, input_size=224)

    # Forward
    with torch.no_grad():
        output = model(input_tensor)
    
    # Parse output
    # p = output[0, 0].item()
    # x = output[0, 1].item()
    # y = output[0, 2].item()
    p = torch.sigmoid(output[0,0]).item()
    x = torch.sigmoid(output[0,1]).item()
    y = torch.sigmoid(output[0,2]).item()
    bw = output[0,3].item()
    bh = output[0,4].item()

    # Map bbox to original frame size
    cx = int(x * w)
    cy = int(y * h)
    box_w = int(bw * w)
    box_h = int(bh * h)
    x1 = max(cx - box_w//2, 0)
    y1 = max(cy - box_h//2, 0)
    x2 = min(cx + box_w//2, w-1)
    y2 = min(cy + box_h//2, h-1)

    # Draw bbox if p > 0.5
    if p > 0.5:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, f"p:{p:.2f} x:{x:.2f} y:{y:.2f} w:{bw:.2f} h:{bh:.2f}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)   

    # Show frame
    cv2.imshow("Fist Detection", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()