import math

from ultralytics import YOLO
import cv2
import numpy as np
import cvzone


CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

model = YOLO('models/yolov8n.pt')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(2, 480)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for result in results:
        bboxes = result.boxes
        for bbox in bboxes:
            # Bounding Boxes
            x1, y1, x2, y2 = bbox.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 22, 200), thickness=3, lineType=1)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img=img, bbox=(x1, y1, w, h))

            # Confidence Score
            conf = math.ceil(bbox.conf[0]*100) / 100

            # Classes
            class_index = int(bbox.cls[0])
            class_name = CLASSES[class_index]
            cvzone.putTextRect(img=img, text=f'{class_name} {conf}', pos=(max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=(64, 64, 64))

    cv2.imshow(winname='Image', mat=img)
    cv2.waitKey(1),