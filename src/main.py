from ultralytics import YOLO
import cv2
import numpy as np
import time 

MATRIX_SIZE = 56
TIMEOUT = 2

# Nạp model YOLO đã train
model = YOLO("src/fist_model/trained.pt")

# Mở webcam (0 là webcam mặc định)
cap = cv2.VideoCapture(0)
last_detect_time = time.time()

# Hàm trích ra frame chứa ảnh vẽ từ frame webcam
def find_sub_frame(draw_matrix):
    total_bits = np.sum(draw_matrix)
    for cur_size in range(MATRIX_SIZE // 4, MATRIX_SIZE + 1, MATRIX_SIZE // 16):
        for i in range(0, MATRIX_SIZE - cur_size + 1, MATRIX_SIZE // 16):
            for j in range(0, MATRIX_SIZE - cur_size + 1, MATRIX_SIZE // 16):
                sub_frame = draw_matrix[i : i+cur_size, j : j + cur_size]
                cur_bits = np.sum(sub_frame)
                if (total_bits == cur_bits):
                    return sub_frame.copy()

# Bắt đầu webcam
detected = False
prev_x, prev_y = None, None
draw_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.uint8)

_, frame = cap.read()
draw_frame = np.zeros_like(frame)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 1 = lật ngang (mirror)
    h_frame, w_frame, _ = frame.shape

    if not ret:
        break

    # Chạy YOLO detect realtime
    results = model(frame, stream=True, imgsz=224) #take out the (x,y) here later

    # Hiển thị kết quả
    display = None
    r = next(iter(results))
    boxes = r.boxes.xywh.cpu().numpy()
    if (len(boxes) > 0):
        box = boxes[0]   
        last_detect_time = time.time()
        detected = True

        x, y, w, h = box
        text = f"({int(x)}, {int(y)})"
        cv2.putText(frame, text, (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        x, y = int(x), int(y)

        if prev_x is None and prev_y is None: 
            prev_x, prev_y = x, y
            
        draw_prev_x = int(prev_x * MATRIX_SIZE / w_frame)
        draw_prev_y = int(prev_y * MATRIX_SIZE / h_frame)
        draw_x = int(x * MATRIX_SIZE / w_frame)
        draw_y = int(y * MATRIX_SIZE / h_frame)

        cv2.line(draw_frame, (prev_x, prev_y), (x, y), (0, 255, 0), 2)
        cv2.line(draw_matrix, (draw_prev_x, draw_prev_y), (draw_x, draw_y), 1, 1)
        prev_x, prev_y = x, y
    
    annotated = r.plot()  # Bbox + label
    display = cv2.addWeighted(annotated, 1, draw_frame, 1, 0) #Lines
    cv2.imshow("YOLO Detection", display)
    
    if (time.time() - last_detect_time > TIMEOUT and detected):
        print("No detection for {TIMEOUT} seconds. Exiting...")
        sub_frame = find_sub_frame(draw_matrix)
        sub_frame *= 255
        sub_frame = cv2.resize(sub_frame, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite("draw.png", sub_frame)
        break

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()