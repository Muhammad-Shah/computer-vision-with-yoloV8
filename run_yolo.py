from ultralytics import YOLO
from utils.download import download_data
import cv2

img_url = ('https://imgs.search.brave.com/MZcVw_uqMRXrrCdbi3wOUlSNxfZBENpSMzqYwLyE28c/rs:fit:500:0:0/g:ce'
           '/aHR0cHM6Ly93d3cu/aXN0b2NrcGhvdG8u/Y29tL3Jlc291cmNl/cy9pbWFnZXMvSG9t/ZVBhZ2UvRm91clBh/Y2svQzItUGhvdG9z'
           '/LWlTdG9jay0xMzU2/MTk3Njk1LmpwZw')

# download_data(data_dir='TestImages', uri=img_url, file_name='test_image.jpg')
model = YOLO('models/yolov8n.pt')

result = model('Data/TestImages/test_image.jpg', show=True)

cv2.waitKey(0)