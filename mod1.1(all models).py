# import the necessary packages
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Loading all models
model1 = YOLO("../Weights/3RIDE.pt")
model2 = YOLO("../Weights/HEL1.pt")
model3 = YOLO("../Weights/yolo11n.pt")
model4 = YOLO("../Weights/LP.pt")
model5 = YOLO("../Weights/PHN1.pt")
model6 = YOLO("../Weights/SBELT.pt")

# Initializing class names for each model
classNames1 = ['no-helmet', 'overloading', 'safe']
classNames2 = ["helmat", "no helmat"]
classNames3 = ["person", "bike", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bird", "cat",
              "dog", "horse", "cow", "backpack", "handbag", "bottle", "cell phone"]  # Selected classes
classNames4 = ["NP"]
classNames5 = ['Distracted']
classNames6 = ['Seat Belt']

# Valid(interested class numbers) indices for classNames3
valid_classes_model3 = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18, 24, 25, 39, 67]

# Initialize previous & new frame time
prev_frame_time = 0
new_frame_time = 0

# change the path to the video file/image file path
cap = cv2.VideoCapture("../Traffic-Violation-Detection-System/pic/hel/2.jpg")

# Infinite loop for processing frames
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break
    
    # Run all models to get results
    results1 = model1(img, stream=True)
    results2 = model2(img, stream=True)
    results3 = model3(img, stream=True)
    results4 = model4(img, stream=True)
    results5 = model5(img, stream=True)
    results6 = model6(img, stream=True)

    # Process results for all models
    for results, classNames in zip([results1, results2, results3, results4, results5, results6], 
                                   [classNames1, classNames2, classNames3, classNames4, classNames5, classNames6]):
        for r in results:# Iterate over results
            boxes = r.boxes  # Get bounding boxes from results
            for box in boxes:  # Iterate over each box
                x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates of the box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
                w, h = x2 - x1, y2 - y1  # Calculate width and height of the box
                cvzone.cornerRect(img, (x1, y1, w, h))  # Draw rectangle on the image
                conf = math.ceil((box.conf[0] * 100)) / 100  # Calculate confidence score
                cls = int(box.cls[0])  # Get class index

                if classNames == classNames3:  # If processing model3
                    if cls in valid_classes_model3:  # If class index is valid
                        display_cls = valid_classes_model3.index(cls)  # Get display class index
                        cvzone.putTextRect(img, f'{classNames[display_cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)  # Display class name and confidence
                else:  # For other models
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)  # Display class name and confidence
                
    fps = 1 / (new_frame_time - prev_frame_time)  # Calculate frames per second (FPS)
    prev_frame_time = new_frame_time  # Update previous frame time
    print(fps)  # Print FPS

    cv2.imshow("Image", img)  # Display the image
    cv2.waitKey(0)  # Wait for key press to proceed to next frame