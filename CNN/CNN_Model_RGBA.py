import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Dataset path দিন
DATASET_DIR = "dataset"  # আপনার dataset folder path

# ImageDataGenerator দিয়ে ডেটা লোড ও normalize করা (RGB/RGBA 4 channel আছে বলে color_mode='rgba')
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(28, 28),
    color_mode='rgba',          # 4 channel এর জন্য 'rgba' দিন
    batch_size=32,
    class_mode='categorical',   # যদি label গুলো one-hot encoded চান
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(28, 28),
    color_mode='rgba',
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# CNN মডেল ডিজাইন (input_shape (28, 28, 4))
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 4)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # স্বয়ংক্রিয় ক্লাস সংখ্যা
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# মডেল ট্রেনিং
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# মডেল সেভ করুন
model.save("cnn_model_rgba.h5")
print("Model saved as cnn_model_rgba.h5")
