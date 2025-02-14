import cv2
import easyocr
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# List of number plates to track
target_number_plates = ["NA13NRU","NVS1VSU","LM13 VCV","EYGI NBG", "KHO522K", 
                        "GXIS OCJ"]

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize video capture
cap = cv2.VideoCapture("s.mp4")  # Update the path

trackers = []

# Function to recognize number plates
def recognize_number_plate(image):
    result = reader.readtext(image)
    number_plates = [item[1] for item in result]
    return number_plates

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    lower_half_frame = frame[height//2:, :]  # Only process lower half of the frame

    if frame_count % 2 == 0:  # Process every 2nd frame
        # Detect vehicles using YOLO
        results = model(lower_half_frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                crop_img = lower_half_frame[y1:y2, x1:x2]

                if crop_img.size == 0:
                    continue

                number_plates = recognize_number_plate(crop_img)

                if any(plate in target_number_plates for plate in number_plates):
                    # Add new tracker for this vehicle
                    tracker = cv2.TrackerCSRT_create()
                    trackers.append(tracker)
                    tracker.init(lower_half_frame, (x1, y1, x2 - x1, y2 - y1))
                    # Draw bounding box
                    cv2.rectangle(lower_half_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(lower_half_frame, number_plates[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        # Update and draw tracked bounding boxes
        for tracker in trackers:
            success, box = tracker.update(lower_half_frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(lower_half_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Increment frame count
    frame_count += 1

    # Place the processed lower half back into the original frame
    frame[height//2:, :] = lower_half_frame

    # Display the frame
    cv2.imshow("Traffic Violation Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
