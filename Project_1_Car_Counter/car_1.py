import math
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from utils.download import download_file

# from sort import Sort

sort_py_file_uri = 'https://raw.githubusercontent.com/abewley/sort/master/sort.py'
download_file(uri=sort_py_file_uri, file_name='sort.py')

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
total_count = []

model = YOLO('../models/yolov8n.pt')
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]


cap = cv2.VideoCapture('../Videos/cars.mp4')
masked_image = cv2.imread('mask.png')
img_graphics = cv2.imread('img.png', cv2.IMREAD_UNCHANGED)

while True:
    success, img = cap.read()
    image_with_region = cv2.bitwise_and(img, masked_image)
    img = cvzone.overlayPNG(img, img_graphics, (0, 0))

    results = model(image_with_region, stream=True)
    detections = np.empty((0, 5))
    for result in results:
        bboxes = result.boxes
        for bbox in bboxes:
            # Bounding Boxes
            x1, y1, x2, y2 = bbox.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 22, 200), thickness=3, lineType=1)
            w, h = x2 - x1, y2 - y1

            # Confidence Score
            conf = math.ceil(bbox.conf[0] * 100) / 100

            # Classes
            class_index = int(bbox.cls[0])
            class_name = CLASSES[class_index]

            if class_name == 'car' or class_name == 'motorcycle' or class_name == 'bus' \
                    or class_name == 'truck' and conf > 0.4:
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    result_tracker = tracker.update(detections)
    cv2.line(img=img, pt1=(limits[0], limits[1]), pt2=(limits[2], limits[3]), color=(0, 0, 255), thickness=3)

    for result in result_tracker:
        x1, y1, x2, y2, detection_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img=img, bbox=(x1, y1, w, h), colorR=(255, 0, 255), l=2, t=1)
        cvzone.putTextRect(img=img,
                           text=f'{int(detection_id)}',
                           pos=(max(0, x1), max(35, y1)),
                           scale=1,
                           thickness=1,
                           offset=2)
        # center of the box
        cx, cy = x1 + w // 2, y1+h//2
        cv2.circle(img=img, center=(cx, cy), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)
        # car pass then count
        if limits[0] < cx < limits[2] and - 15 < cy < limits[3] + 15:
            if total_count.count(detection_id) == 0:
                total_count.append(detection_id)
                cv2.line(img=img, pt1=(limits[0], limits[1]), pt2=(limits[2], limits[3]), color=(0, 255, 0),
                         thickness=3)

    cvzone.putTextRect(img=img, text=f'{len(total_count)}', pos=(255, 100), scale=3, thickness=2,)
    cv2.imshow(winname='Image', mat=img)
    # cv2.imshow(winname='Image', mat=image_with_region)
    cv2.waitKey(0)
