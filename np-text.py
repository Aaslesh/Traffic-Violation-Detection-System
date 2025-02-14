import cv2
import easyocr
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to recognize number plates
def recognize_number_plate(image):
    result = reader.readtext(image)
    number_plates = [item[1] for item in result]
    return number_plates

# Function to process a given frame or image and detect number plates
def process_image(image, model, reader):
    number_plates_list = []
    
    # Detect vehicles using YOLO
    results = model(image)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            crop_img = image[y1:y2, x1:x2]
            number_plates = recognize_number_plate(crop_img)
            number_plates_list.extend(number_plates)
    
    return number_plates_list

# Function to process a video and detect number plates
def process_video(video_path, model, reader):
    cap = cv2.VideoCapture(video_path)
    all_number_plates = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        number_plates_in_frame = process_image(frame, model, reader)
        all_number_plates.extend(number_plates_in_frame)
        
        # Display the frame
        cv2.imshow("Traffic Violation Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return all_number_plates

def main(video_path=None, image_path=None):
    detected_number_plates = []
    
    # Try processing video if video_path is provided and valid
    if video_path:
        try:
            detected_number_plates = process_video(video_path, model, reader)
        except Exception as e:
            print(f"Error processing video: {e}")
    
    # If video processing failed or not provided, process the image
    if not detected_number_plates and image_path:
        try:
            image = cv2.imread(image_path)
            detected_number_plates = process_image(image, model, reader)
        except Exception as e:
            print(f"Error processing image: {e}")
    
    return detected_number_plates

# Paths to video and image
video_path = "s_ini.mp4"  # Update the path
image_path = "../mini1/pic/np/2.jpg"  # Update the path

# Detect number plates
detected_number_plates = main(video_path, image_path)

# Print detected number plates
print("Detected Number Plates:", detected_number_plates)
