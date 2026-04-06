"""
    Train YOLO with Ultralytics (yolo8n)

    For Text Detection on ID Card
"""

from ultralytics import YOLO


DATA_PATH = "/home/truongkhanh/work/learn-by-me/cv/01-ocr-id-card/dataset/SROIE2019/data.yaml"


def train():
    model = YOLO('ckpt/detect_text/yolov8n.pt') 

    # 2. Bắt đầu huấn luyện
    results = model.train(
        data=DATA_PATH,    # Đường dẫn file yaml bạn vừa tạo
        epochs=100,          # Số vòng lặp huấn luyện
        imgsz=640,           # Kích thước ảnh đầu vào
        batch=8,            # Số lượng ảnh mỗi đợt (tùy vào VRAM của GPU)
        device=0,            # Chạy trên GPU (0) hoặc CPU ('cpu')
        name='my_yolo_model' # Tên thư mục lưu kết quả
    )
    

def check():
    model = YOLO('runs/detect/my_yolo_model2/weights/best.pt') 

    predictions = model("result/image6.png")
    predictions[0].show()

def predict(path):
    model = YOLO('runs/detect/my_yolo_model2/weights/best.pt') 

    predictions = model(path)
    return predictions


if __name__ == "__main__":
    check()