import cv2
import torch
import numpy as np
from src.fist_model.model.model import FistDetection

# --- Load model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FistDetection().to(device)
model.load_state_dict(torch.load("src/fist_model/trained.pth", map_location=device))
model.eval()

# --- Webcam ---
cap = cv2.VideoCapture(0)
INPUT_SIZE = 224

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize & preprocess
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        output = model(input_tensor)
        output = output[0].cpu().numpy()  # (5,)
    
    p, x, y, w, h = output
    if p > 0.5:
        # Chuyển bbox về kích thước frame gốc
        x_center = int(x * frame.shape[1])
        y_center = int(y * frame.shape[0])
        box_w = int(w * frame.shape[1])
        box_h = int(h * frame.shape[0])

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # Vẽ bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"P: {p:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Fist Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
