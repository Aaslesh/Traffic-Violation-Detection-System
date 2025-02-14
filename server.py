from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
import math
import time
import cvzone
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize YOLO models
model1 = YOLO("../Weights/3RIDE.pt")
model2 = YOLO("../Weights/HEL1.pt")
model3 = YOLO("../Weights/yolo11n.pt")
model4 = YOLO("../Weights/LP.pt")
model5 = YOLO("../Weights/PHN1.pt")
model6 = YOLO("../Weights/SBELT.pt")

classNames1 = ['no-helmet', 'overloading', 'safe']
classNames2 = ["helmet", "no helmet"]
classNames3 = ["person", "bike", "car", "motorbike", "bus", "truck", "traffic light", "cell phone"]
classNames4 = ["NP"]
classNames5 = ['Distracted']
classNames6 = ['Seat Belt']
valid_classes_model3 = [0, 1, 2, 3, 5, 6, 7, 67]

video_path = None
location = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    global video_path
    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No file part'
        file = request.files['video']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            return redirect(url_for('index'))
    return render_template('main.html', video_path=video_path)

def generate_frames():
    global video_path
    if not video_path:
        print("No video path set.")
        return

    cap = cv2.VideoCapture(video_path)
    prev_frame_time = 0

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break

        new_frame_time = time.time()

        results1 = model1(img, stream=True)
        results2 = model2(img, stream=True)
        results3 = model3(img, stream=True)
        results4 = model4(img, stream=True)
        results5 = model5(img, stream=True)
        results6 = model6(img, stream=True)

        for results, classNames in zip([results1, results2, results3, results4, results5, results6],
                                       [classNames1, classNames2, classNames3, classNames4, classNames5, classNames6]):
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    if classNames == classNames3 and cls in valid_classes_model3:
                        display_cls = valid_classes_model3.index(cls)
                        cvzone.putTextRect(img, f'{classNames[display_cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                    elif classNames != classNames3:
                        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

    cap.release()

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
