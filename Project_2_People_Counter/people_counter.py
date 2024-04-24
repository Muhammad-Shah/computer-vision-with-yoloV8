import math
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
# from utils.download import download_file
from sort import Sort

sort_py_file_uri = 'https://raw.githubusercontent.com/abewley/sort/master/sort.py'
# download_file(uri=sort_py_file_uri, file_name='sort.py')

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
total_count_up = []
total_count_down = []

model = YOLO('../models/yolov8n.pt')
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsUp = [100, 200, 300, 198] # [x1, y1, x2, y2]
limitsDown = [500, 400, 650, 350]


cap = cv2.VideoCapture('../Videos/people.mp4')
masked_image = cv2.imread('mask.png')
graphics_img = cv2.imread('graphics.png', cv2.IMREAD_UNCHANGED)

while True:
    success, img = cap.read()
    image_with_region = cv2.bitwise_and(img, masked_image)
    img = cvzone.overlayPNG(imgBack=img, imgFront=graphics_img, pos=(720, 260))
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

            if class_name == 'person' and conf > 0.4:
                current_bbox = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_bbox))

    # take detections/bboxs and give an id to each bbox
    print(detections.shape)
    result_tracker = tracker.update(detections)
    cv2.line(img=img, pt1=(limitsUp[0], limitsUp[1]), pt2=(limitsUp[2], limitsUp[3]), color=(0, 0, 255), thickness=3)
    cv2.line(img=img, pt1=(limitsDown[0], limitsDown[1]), pt2=(limitsDown[2], limitsDown[3]), color=(0, 0, 255), thickness=3)

    for result in result_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img=img, bbox=(x1, y1, w, h), colorR=(255, 0, 255), l=2, t=1)
        cvzone.putTextRect(img=img,
                           text=f'{int(id)}',
                           pos=(max(0, x1), max(35, y1)),
                           scale=1,
                           thickness=1,
                           offset=2)
        # center of the box
        cx, cy = x1 + w // 2, y1+h//2
        cv2.circle(img=img, center=(cx, cy), radius=10, color=(255, 0, 255), thickness=cv2.FILLED)
        # person pass then count
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[3] + 15:
            if total_count_up.count(id) == 0:
                total_count_up.append(id)
                cv2.line(img=img, pt1=(limitsUp[0], limitsUp[1]), pt2=(limitsUp[2], limitsUp[3]), color=(0, 255, 0),
                         thickness=3)
        # person pass then count
        if limitsDown[0] < cx < limitsDown[2] and limitsUp[1] - 15 < cy < limitsDown[3] + 15:
            if total_count_down.count(id) == 0:
                total_count_down.append(id)
                cv2.line(img=img, pt1=(limitsDown[0], limitsDown[1]), pt2=(limitsDown[2], limitsDown[3]), color=(0, 255, 0),
                         thickness=3)

    cv2.putText(img=img, org=[920, 345], thickness=7, text=str(int(len(total_count_up))), color=(139, 195, 75), fontScale=5,
                fontFace=cv2.FONT_HERSHEY_PLAIN)
    cv2.putText(img=img, text=str(int(len(total_count_down))), org=(1191, 345), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5,
                color=(50, 50, 230), thickness=7)
    cv2.imshow(winname='Image', mat=img)
    cv2.waitKey(0)
