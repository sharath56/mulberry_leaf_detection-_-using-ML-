import cv2
import torch
import sys
import time
from models.experimental import attempt_load
from utils.general import non_max_suppression

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Define the paths to the model files
model_paths = {
    'model1': 'C:/Users/sharath.narayanaswam/Desktop/object_detection/yolov5/models/BT/mulberry.pt'
}

class_names = {

    'model1': {
        0: 'Mulberry_leaf',
        1: 'weed'
    }}

# Load the models
models = {}
for name, path in model_paths.items():
    models[name] = attempt_load(path).to(device).eval().fuse()

# Define the object detection function
def detect_objects(frame, conf_thresh=0.5, iou_thresh=0.45):
    # Preprocess the input image
    img = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
    img = img.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img.to(device)

    # Perform object detection with each model
    results = []
    for name, model in models.items():
        # Get the model's output
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thresh, iou_thresh)

        # Add the detections to the results list
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = det[:, :4].clone().int()
                for d in det:
                    results.append({
                        'name': name,
                        'class': int(d[5]),
                        'confidence': float(d[4]),
                        'bbox': [float(x) for x in d[:4]]
                    })

    return results
# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = detect_objects(frame)

    for det in results:
        bbox = det['bbox']
        class_name = class_names[det['name']][det['class']]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(frame, f"{class_name}: {det['confidence']:.2f}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows()
