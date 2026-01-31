from sort import *
from ultralytics import YOLO
import numpy as np
from util import get_car
import cv2

# model untuk deteksi mobil
model_mobil = YOLO(r"D:\Belajar FR\Youtube_2\plate-detection\yolo11n.pt")

# model untuk deteksi plat nomor
model_plat = YOLO(r"D:\Belajar FR\Youtube_2\plate-detection\model_plate_detection.pt")


cap =cv2.VideoCapture(r"D:\Belajar FR\Youtube_2\plate-detection\sample.mp4")

tracker = Sort()

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

vehicles = [2, 3, 5, 7]

frame_number = -1
succes = True
while succes:
    frame_number += 1
    succes, frame = cap.read()
    if succes and frame_number < 1:
        detection = model_mobil(frame)[0]


        detection_2 = []
        box_2 = []
        for box in detection.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box            
            if int(cls_id) in vehicles:
                detection_2.append([x1, y1, x2, y2, conf]) # informasi bbox mobil               

    tracks_id = tracker.update(np.array(detection_2))# berisikan informasi tentang bbox mobil
    
    plat = model_plat(frame)[0]
    for box_plat in plat.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = box_plat 

        # koneksikan plat ke mobil
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(box_plat, tracks_id)

        # crop plat
        crop_plat = frame[int(y1):int(y2), int(x1):int(x2), :]

        # proses plat
        crop_plat_gray = cv2.cvtColor(crop_plat, cv2.COLOR_BGR2GRAY)
        _, crop_plat_thresh = cv2.threshold(crop_plat_gray, 64, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow("orignal_crop", crop_plat)
        cv2.imshow("threshold", crop_plat_thresh)
    
    