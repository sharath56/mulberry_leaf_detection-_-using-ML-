import cv2

# Read the live video stream
cap = cv2.VideoCapture(0)

# Create a machine learning model using a pre-trained CNN
model = cv2.dnn.readNetFromTensorflow('model.pb')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply the model to classify the frame
    blob = cv2.dnn.blobFromImage(thresh, 1, (224, 224), (104, 117, 123), swapRB=True, crop=False)
    model.setInput(blob)
    output = model.forward()

    # Overlay the classification results on the frame
    if output[0][1] > output[0][0]:
        cv2.putText(frame, 'Mulberry Leaf', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No Mulberry Leaf', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
