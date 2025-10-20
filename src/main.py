from ultralytics import YOLO
import cv2
import numpy as np
import time 
from src.draw_model.model.model import DrawModel
from src.draw_model.config.constant import NUM_CLASSES
import torch

MATRIX_SIZE = 56
TIMEOUT = 2
TIME_SLEEP = 5

# Nạp model YOLO đã train
model = YOLO("src/fist_model/trained.pt")

# Mở webcam (0 là webcam mặc định)
cap = cv2.VideoCapture(0)
last_detect_time = time.time()

# Hàm trích ra frame chứa ảnh vẽ từ frame webcam
def find_sub_frame(draw_matrix):
    total_bits = np.sum(draw_matrix)
    H, W = draw_matrix.shape
    a = min(H, W) 

    for cur_size in range(max(28, a // 8), a + 1, a // 16):
        for i in range(0, H - cur_size + 1, cur_size // 3):
            for j in range(0, W - cur_size + 1, cur_size // 3):
                cur_frame = draw_matrix[i:i+cur_size, j:j+cur_size]
                cur_bits = np.sum(cur_frame)
                if (cur_bits == total_bits):
                    print("Found sub frame!!!!")
                    
                    #Move the object to the center
                    step = cur_size // 5
                    for _i in range(i, i + cur_size, step):
                        cur_frame = draw_matrix[_i:_i+cur_size, j:j+cur_size]
                        cur_bits = np.sum(cur_frame)
                        if (cur_bits < total_bits):
                            i = (i + _i - step) // 2
                            break

                    for _j in range(j, j + cur_size, step):
                        cur_frame = draw_matrix[i:i+cur_size, _j:_j+cur_size]
                        cur_bits = np.sum(cur_frame)
                        if (cur_bits < total_bits):
                            j = (j + _j - step) // 2
                            break
                    
                    cur_frame = draw_matrix[i:i+cur_size, j:j+cur_size]
                    cur_frame = cv2.resize(cur_frame, (28, 28), interpolation=cv2.INTER_AREA)

                    #find the positive median
                    contain = cur_frame > 0
                    med = np.median(cur_frame[contain])
                    cur_frame /= (med) # nearly normalize
                    return cur_frame
                
    print("No sub frame found!")
    return None    #actually, this would never happen

# Bắt đầu webcam
detected = False
prev_x, prev_y = None, None

_, frame = cap.read()
draw_frame = np.zeros_like(frame) #for display on webcam
draw_model = DrawModel(num_classes=NUM_CLASSES)
draw_matrix = np.zeros(frame.shape[:2]) #for classification

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 1 = lật ngang (mirror)
    h_frame, w_frame, _ = frame.shape

    if not ret:
        break

    # Chạy YOLO detect realtime
    results = model(frame, stream=False, imgsz=224) #take out the (x,y) here later

    # Hiển thị kết quả
    display = None
    r = results[0]
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

        cv2.line(draw_frame, (prev_x, prev_y), (x, y), (0, 255, 0), 2)
        cv2.line(draw_matrix, (prev_x, prev_y), (x, y), 255, 1)
        prev_x, prev_y = x, y
    
    annotated = r.plot()  # Bbox + label
    display = cv2.addWeighted(annotated, 1, draw_frame, 1, 0) #Lines
    cv2.imshow("YOLO Detection", display)
    
    if (time.time() - last_detect_time > TIMEOUT and detected):
        print("No detection for {TIMEOUT} seconds. Exiting...")
        sub_frame = find_sub_frame(draw_matrix)
        sub_frame *= 255
        cv2.imwrite("draw.png", sub_frame); print("Save picture successfully!")

        sub_frame = torch.from_numpy(sub_frame).float() / 255.0
        sub_frame = sub_frame.reshape(1, -1, 28, 28)
        y_pred = draw_model.predict(sub_frame)
        text = f"Image name: {y_pred[0]}"
        print(text)
        cv2.putText(display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("YOLO Detection", display)
        cv2.waitKey(TIME_SLEEP)  # chờ 3 giây để xem kết quả
        break

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()