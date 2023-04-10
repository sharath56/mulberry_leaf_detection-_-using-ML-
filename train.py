import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the image size and batch size
img_size = 224
batch_size = 32

# Define the paths to the image folders
mulberry_dir = "mulberry/"
non_mulberry_dir = "non-mulberry/"

# Create a list of all the image paths
mulberry_paths = [os.path.join(mulberry_dir, f) for f in os.listdir(mulberry_dir)]
non_mulberry_paths = [os.path.join(non_mulberry_dir, f) for f in os.listdir(non_mulberry_dir)]
image_paths = mulberry_paths + non_mulberry_paths

# Create the labels for the images
mulberry_labels = [1] * len(mulberry_paths)
non_mulberry_labels = [0] * len(non_mulberry_paths)
labels = mulberry_labels + non_mulberry_labels

# Split the data into training and testing sets
X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Define a function to load and preprocess an image
def load_preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img

# Define a function to create the dataset
def create_dataset(X_paths, y):
    X = []
    for path in X_paths:
        img = load_preprocess_image(path)
        X.append(img)
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

# Create the training and testing datasets
train_ds = create_dataset(X_train_paths, y_train)
test_ds = create_dataset(X_test_paths, y_test)

# Create the CNN model
import tensorflow as tf

# Define the input shape
input_shape = (224, 224, 3)

# Create the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(train_ds, epochs=10, validation_data=test_ds)

# Save the model
model.save("mulberry_leaf_detector.h5")

# Load the saved model
loaded_model = tf.keras.models.load_model("mulberry_leaf_detector.h5")

# Use the model to make predictions on new data
prediction = loaded_model.predict(new_data)

