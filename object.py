import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load your custom dataset (replace with your paths)
data_dir = "imageyololabel/images"
class_names = sorted(os.listdir(data_dir))  # Get class names from folders
num_classes = len(class_names)

# 2. Create dataset using image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32)

# 3. Define model (updated with softmax for proper classification)
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),  # Added extra layer
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Changed to softmax
])

# 4. Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Simplified loss
    metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10)

# 5. Save model
model.save('cnn_classifier.h5')