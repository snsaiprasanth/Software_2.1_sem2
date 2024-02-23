# from IPython import display
import cv2

# from supervision.tools.detections import Detections, BoxAnnotator
from os import listdir
from os.path import isfile, join
from typing import List
import ultralytics

ultralytics.checks()
from ultralytics import YOLO

import numpy as np

# Import all images in the path
# SOURCE_PATH = r"D:\2024_WORKSHOP_IAAC\3D_model\input\iaac_building_01"
SOURCE_PATH = r"D:\2024_WORKSHOP_IAAC\3D_model\input\iteration_02\frames"
output_path = r"D:\2024_WORKSHOP_IAAC\3D_model\input\iteration_02\detected_frames"
onlyfiles = [f for f in listdir(SOURCE_PATH) if isfile(join(SOURCE_PATH, f))]


# settings
MODEL = r"D:\2024_WORKSHOP_IAAC\custom_train\iteration_02\100epochs\weights\best.pt"


model = YOLO(MODEL)
model.fuse()
#* In case that you don't want to detect all classes
# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]

# loop over video frames
for frame in onlyfiles: #tqdm(generator, total=video_info.total_frames):
    # model prediction on single frame and conversion to supervision Detections
    name_text = frame
    frame = cv2.imread(f"{SOURCE_PATH}/{frame}")
    width, height = frame.shape[:2]
    print(width, height)

    # Detect classes
    results = model(frame)

    # Extract xyxy, confidence, class_id
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int)

    test1 = xyxy[0]

    gray_frame = frame.copy()
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
    # Add three channels to the grayscale image
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    i = 0
    for bbox in test1:
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # cropped_image = cropped_image[:, :, 2]  # BGR channel order is 0: Blue, 1: Green, 2: Red
        red_crop = cropped_image.copy()
        
        # Purple 
        if class_id[i] == 0:
            # set green channel to 0
            red_crop[:, :, 1] = 0

        # Green 
        if class_id[i] == 1:
            # set green channel to 0
            red_crop[:, :, 0] = 0
            red_crop[:, :, 2] = 0

        gray_frame[int(bbox[1]):int(bbox[1])+red_crop.shape[0], int(bbox[0]):int(bbox[0])+red_crop.shape[1]] = red_crop
        
        # Increase iteration counter
        i += 1





    # print(red_crop.shape)
    # cv2.imwrite('red_crop.jpg', red_crop)
    # print(f"{SOURCE_PATH}{frame}")
    cv2.imwrite(f"{output_path}/{name_text}", gray_frame)
    
