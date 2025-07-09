import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

# 1. Image configuration
IMAGE_SIZE = (28, 28)
BATCH_SIZE = 32
DATASET_DIR = "dataset"  # আপনার ডেটাসেট ফোল্ডারের নাম

# 2. Data loading and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

# 3. CNN Model Architecture
model = Sequential([
    Input(shape=(28, 28, 1)),  # ✅ input_shape এর warning বন্ধ
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # ✅ 10 ক্লাস: 0-9
])

# 4. Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    verbose=2  # ✅ Short summary log
)

# 6. Save the model
model.save("cnn_model_digits_0_to_9.h5")
print("✅ Model saved as cnn_model_digits_0_to_9.h5")
