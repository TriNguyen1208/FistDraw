import cv2
import torch
import torch.nn as nn
from src.build_yolo.model import TinyYOLO  # import class TinyYOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = TinyYOLO().to(device)
model.load_state_dict(torch.load("src/build_yolo/tinyyolo_from_scratch.pth", map_location=device))
model.eval()

# Webcam
cap = cv2.VideoCapture(0)
img_size = 224

# Preprocessing transform
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
        output = model(img)  # [1,5]
        output = output[0].cpu()  # [5]

        x, y, w, h, conf = output.tolist()

        h_frame, w_frame = frame.shape[:2]
        cx = int(x * w_frame)
        cy = int(y * h_frame)
        bw = int(w * w_frame)
        bh = int(h * h_frame)
        
        x1 = int(cx - bw/2)
        y1 = int(cy - bh/2)
        x2 = int(cx + bw/2)
        y2 = int(cy + bh/2)

        # In ra giá trị bbox + confidence
        text = f"x:{x:.2f} y:{y:.2f} w:{w:.2f} h:{h:.2f} conf:{conf:.2f}"
        cv2.putText(frame, text, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        if conf > 0.5:  # threshold confidence
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            
        cv2.imshow("TinyYOLO Webcam", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
