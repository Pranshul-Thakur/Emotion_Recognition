import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model

# Load the pre-trained emotion recognition model
emotion_model_path = r'C:\Users\LENOVO\Desktop\Emotion_Recognition\emotion_model.hdf5'
emotion_model = load_model(emotion_model_path)

# Load the pre-trained face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the facial landmarks predictor from dlib
predictor_path = r'C:\Users\LENOVO\Desktop\Emotion_Recognition\shape_predictor_68_face_landmarks (1).dat'
landmark_predictor = dlib.shape_predictor(predictor_path)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open the camera
cap = cv2.VideoCapture(0)

# Create a separate window for emotion percentages
cv2.namedWindow('Emotion Percentages', cv2.WINDOW_NORMAL)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face to match the input size of the emotion model
        face_roi_resized = cv2.resize(face_roi, (64, 64))

        # Normalize pixel values
        face_roi_normalized = face_roi_resized / 255.0

        # Reshape the face to match the input shape of the model
        face_input = np.reshape(face_roi_normalized, (1, 64, 64, 1))

        # Make emotion prediction
        emotion_predictions = emotion_model.predict(face_input)
        predicted_emotion = emotion_labels[np.argmax(emotion_predictions)]

        # Draw bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {predicted_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display emotion percentages in a separate window
        emotion_percentages = [f'{label}: {round(prob * 100, 2)}%' for label, prob in zip(emotion_labels, emotion_predictions[0])]
        emotion_text = ', '.join(emotion_percentages)
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Detect facial landmarks
        landmarks = landmark_predictor(frame, dlib.rectangle(x, y, x + w, y + h))

        # Draw facial landmarks with fine lines and color gradient
        for i in range(68):
            x_landmark, y_landmark = landmarks.part(i).x, landmarks.part(i).y
            color = (0, int(255 * i / 68), int(255 * (68 - i) / 68))
            cv2.circle(frame, (x_landmark, y_landmark), 2, color, -1)

        # Draw lines with anti-aliased effect
        lines = [[*range(0, 17)], [*range(17, 22)], [*range(22, 27)], [*range(27, 31)], [*range(31, 36)],
                 [*range(36, 42)], [*range(42, 48)], [*range(48, 60)], [*range(60, 68)]]
        for line in lines:
            for i in range(len(line) - 1):
                x1, y1 = landmarks.part(line[i]).x, landmarks.part(line[i]).y
                x2, y2 = landmarks.part(line[i + 1]).x, landmarks.part(line[i + 1]).y
                cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Overlay the frame with reduced opacity
    cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
