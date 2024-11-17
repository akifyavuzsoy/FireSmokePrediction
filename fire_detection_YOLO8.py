import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO
from IPython.display import display, Image
from IPython.display import Image as show_image

class FireSmokeDetectionYOLO():
    def __init__(self):
        self.model = None
        self.result = None
        self.test_result = None

    def build_model(self):
        self.model = YOLO('yolov8n.pt')

        self.result = self.model.train(data='datasets/data.yaml', epochs=5, imgsz=640)

    def test_model(self):
        self.test_result = self.model.predict(source='test_img_2.png', save=True)

    def save_model(self):
        self.model.save('my_model.pt')

    def show_result(self):
        img = cv2.imread('runs/detect/train2/first_test_1.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Dataset oluştur
    YOLO_Model = FireSmokeDetectionYOLO()
    
    YOLO_Model.build_model()
    YOLO_Model.test_model()
    YOLO_Model.save_model()
    YOLO_Model.show_result()
    

    
