import os
!pip install kaggle
import os
# Create the Kaggle directory if it doesn't exist
os.makedirs('/root/.kaggle', exist_ok=True)
# Create the Kaggle directory if it doesn't exist
kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)

# Copy the kaggle.json file to the Kaggle directory
import shutil
src_path = r"C:\Users\sradd\Downloads\kaggle (1).json"
dst_path = os.path.join(kaggle_dir, 'kaggle.json')
shutil.copy(src_path, dst_path)
from kaggle.api.kaggle_api_extended import KaggleApi
api.dataset_download_files('msambare/fer2013', path='./fer2013', unzip=True)
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define the path to the train and test folders
train_dir = './fer2013/train'
test_dir = './fer2013/test'

# Define the emotions
emotions = os.listdir(train_dir)

# Load the training images
train_images = []
train_labels = []
for emotion in emotions:
    emotion_dir = os.path.join(train_dir, emotion)
    for filename in os.listdir(emotion_dir):
        img = Image.open(os.path.join(emotion_dir, filename)).convert('L')  # Convert to grayscale
        img = img.resize((48, 48))  # Resize the image to 48x48
        img = np.array(img) / 255.0  # Normalize the image
        train_images.append(img)
        train_labels.append(emotions.index(emotion))

# Load the testing images
test_images = []
test_labels = []
for emotion in emotions:
    emotion_dir = os.path.join(test_dir, emotion)
    for filename in os.listdir(emotion_dir):
        img = Image.open(os.path.join(emotion_dir, filename)).convert('L')  # Convert to grayscale
        img = img.resize((48, 48))  # Resize the image to 48x48
        img = np.array(img) / 255.0  # Normalize the image
        test_images.append(img)
        test_labels.append(emotions.index(emotion))

# Convert the lists to numpy arrays
train_images = np.array(train_images).reshape((-1, 48, 48, 1))
train_labels = to_categorical(train_labels)
test_images = np.array(test_images).reshape((-1, 48, 48, 1))
test_labels = to_categorical(test_labels)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(emotions), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {accuracy:.2f}')
model.save('emotion_model.h5')

import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('emotion_model.h5')

# Define the emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Loop through the faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize the face ROI to 48x48
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Normalize the face ROI
        face_roi = face_roi / 255.0
        
        # Reshape the face ROI to (1, 48, 48, 1)
        face_roi = face_roi.reshape((1, 48, 48, 1))
        
        # Predict the emotion
        prediction = model.predict(face_roi)
        emotion_index = np.argmax(prediction)
        emotion = emotions[emotion_index]
        
        # Display the emotion
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
