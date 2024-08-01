from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
from flask_cors import CORS

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Load pretrained YOLOv8n model
model = YOLO('yolov8x.pt')


@app.route('/process_traffic_status', methods=['POST'])
def process_traffic_status():
    # Nhận hình ảnh từ request
    img_data = request.files['image'].read()
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Thực hiện inference
    results = model(img, conf=0.03)
    
    # Tạo từ điển để lưu số lượng các vật thể của mỗi loại
    object_counts = {}
    
    # Lặp qua các kết quả và đếm số lượng các vật thể của mỗi loại
    for result in results:
        # Lấy các hộp giới hạn và nhãn tương ứng từ kết quả
        boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
        labels = result.names
        
        # Duyệt qua từng hộp giới hạn và nhãn
        for box in boxes:  # Iterate over boxes
            class_id = int(box.cls[0])  # Get class ID
            class_name = labels[class_id]  # Get class name using the class ID
            
            # Tăng số lượng của loại vật thể trong từ điển
            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1
                
                
    print(object_counts)
    totalobject = 0
    if 'car' in object_counts:
        totalobject += object_counts['car'] * 4
    if 'truck' in object_counts:
        totalobject += object_counts['truck'] * 8
    if 'bus' in object_counts:
        totalobject += object_counts['bus'] * 12
    if 'bicycle' in object_counts:
        totalobject += object_counts['bicycle'] * 1
    
    if 'motorcycle' in object_counts and 'person' in object_counts:
        totalobject += object_counts['motorcycle'] if object_counts['motorcycle'] > object_counts['person'] else object_counts['person']
    
    traffic_status = ''
    if totalobject < 60:
        traffic_status = 'Low'
    elif totalobject < 120:
        traffic_status = 'Medium'
    else:
        traffic_status = 'High'
    return jsonify({'traffic_status': traffic_status})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)