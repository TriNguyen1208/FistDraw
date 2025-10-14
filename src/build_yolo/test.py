import cv2
import torch
import torch.nn as nn
from src.build_yolo.model import YOLOv1Tiny

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = YOLOv1Tiny().to(device)
model.load_state_dict(torch.load("src/build_yolo/tinyyolo_from_scratch.pth", map_location=device))
model.eval()

# Webcam
cap = cv2.VideoCapture(0)
img_size = 224

import torchvision.transforms as T
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((img_size,img_size)),
    T.ToTensor(),
])

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = transform(frame).unsqueeze(0).to(device)  # [1,3,H,W]
        output = model(img)[0]  # [S, S, 5]

        # --- Lấy cell có confidence cao nhất ---
        confs = output[...,0]
        cell_idx = torch.argmax(confs)
        cy, cx = divmod(cell_idx.item(), output.shape[1])
        p, x_cell, y_cell, w, h = output[cy, cx].tolist()

        # Chuyển về tọa độ ảnh gốc
        S = output.shape[0]
        h_frame, w_frame = frame.shape[:2]
        cx_img = int((cx + x_cell) / S * w_frame)
        cy_img = int((cy + y_cell) / S * h_frame)
        bw = int(w * w_frame)
        bh = int(h * h_frame)

        x1 = int(cx_img - bw/2)
        y1 = int(cy_img - bh/2)
        x2 = int(cx_img + bw/2)
        y2 = int(cy_img + bh/2)

        # Vẽ bbox + text
        text = f"x:{x_cell:.2f} y:{y_cell:.2f} w:{w:.2f} h:{h:.2f} conf:{p:.2f}"
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        if p > 0.5:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            
        cv2.imshow("TinyYOLO Webcam", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
