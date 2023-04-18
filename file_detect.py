import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('./wa7ed_model.h5')

# Load the eye classifier
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Start the video feed
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes using the classifier
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    # Loop through each eye and apply the model
    for (x, y, w, h) in eyes:
        # Extract the eye from the frame
        eye = frame[y:y+h, x:x+w]

        # Resize the eye to (224, 224)
        eye = cv2.resize(eye, (224, 224))

        # Convert the eye to a numpy array
        eye = np.array(eye).reshape((-1, 224, 224, 3))

        # Predict on the eye
        prediction = model.predict(eye)

        # Print the prediction
        if prediction ==1:
            print('The eye is drowsy')
        else:
            print('The eye is not drowsy')

        # Draw a rectangle around the eye
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Wait for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video feed
cap.release()
cv2.destroyAllWindows()
