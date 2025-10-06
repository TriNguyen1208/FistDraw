import cv2, torch, numpy as np
import src.fist_model.model.model as md

device = "cuda" if torch.cuda.is_available() else "cpu"
model = md.FistDetection().to(device)
model.load_state_dict(torch.load("src/fist_model/trained.pth", map_location=device))
model.eval()

INPUT_SIZE = 224  # chỉnh theo model của bạn nếu khác

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    # mean/std nếu có:
    # mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
    # img = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    return tensor

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Can't open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]

    with torch.no_grad():
        out = model(preprocess(frame)).squeeze(0).cpu().numpy()

    # 🔍 In kết quả raw từ model
    print(f"Output: {out}")

    # Hoặc in tách từng giá trị
    p, bx, by, bw, bh = out
    print(f"p={p:.4f}, bx={bx:.4f}, by={by:.4f}, bw={bw:.4f}, bh={bh:.4f}")

    cv2.putText(frame, f"p={p:.3f} bx={bx:.3f} by={by:.3f} bw={bw:.3f} bh={bh:.3f}",
                (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if p > 0.5:
        x1 = int((bx - bw/2) * W); y1 = int((by - bh/2) * H)
        x2 = int((bx + bw/2) * W); y2 = int((by + bh/2) * H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        tx1 = int(bx * W); ty1 = int(by * H)
        tx2 = int((bx + bw) * W); ty2 = int((by + bh) * H)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0,0,255), 2)

    cv2.imshow("Fist detect (green=center, red=tl). Press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
