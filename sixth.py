import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
import sys

# Set the default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set the path to your dataset file
dataset_path = 'C:/Users/ajays/OneDrive/Desktop/online/assign4/df_file.csv'

# Load the dataset using pandas
df = pd.read_csv(dataset_path)

# Set the dimensions of your input images
height = 100  # Set the height of input images
width = 100   # Set the width of input images

# Set other parameters
batch_size = 32
num_classes = 10  # Number of classes (assuming you have 10 classes)

# Step 3: CNN Architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Step 4: Preprocessing
# Perform any necessary preprocessing steps here

# Step 6: Model Training
# Compile the model
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming you have a dataset of images
# You need to replace this with the actual image data and labels
# Example: X_train, y_train, X_val, y_val = load_data()

# Dummy data generation
X_train = tf.random.normal((100, height, width, 3))
y_train = tf.random.uniform((100,), maxval=num_classes, dtype=tf.int32)
X_val = tf.random.normal((50, height, width, 3))
y_val = tf.random.uniform((50,), maxval=num_classes, dtype=tf.int32)

# One-hot encode the target
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model weights
# model.save_weights('saved_model.h5')
model.save_weights('saved_model.weights.h5')

# Load an example image using OpenCV
image_path = 'C:/Users/ajays/OneDrive/Desktop/online/assign4/text.jpeg'  # Replace with the path to your image

image = cv2.imread(image_path)

# Preprocess the image
resized_image = cv2.resize(image, (height, width))  # Resize the image to match model's input size
normalized_image = resized_image / 255.0  # Normalize pixel values (assuming you're using normalization)

# Expand dimensions to match model's input shape
input_image = np.expand_dims(normalized_image, axis=0)

# Predict text from the image
predicted_text = model.predict(input_image)

# Print the predicted text
print(predicted_text)
