from ultralytics import YOLO
import cv2

# Nạp model YOLO đã train
model = YOLO("src/temp_fist_model/best.pt")

# Mở webcam (0 là webcam mặc định)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 1 = lật ngang (mirror)

    if not ret:
        break

    # Chạy YOLO detect realtime
    results = model(frame, stream=True, imgsz=224) #take out the (x,y) here later

    # Hiển thị kết quả
    for r in results:
        boxes = r.boxes.xywh.cpu().numpy()
        for box in boxes:
            x, y, w, h = box
            text = f"({int(x)}, {int(y)})"
            cv2.putText(frame, text, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        annotated = r.plot()  # Vẽ bbox và label lên ảnh
        cv2.imshow("YOLO Detection", annotated)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()