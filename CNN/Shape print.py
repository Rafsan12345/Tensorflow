import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Image Data Generator সেট করুন
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 2. Training Data লোড করুন (shuffle=False যাতে label ঠিক থাকে)
train_generator = datagen.flow_from_directory(
    'dataset',               # ✅ আপনার ফোল্ডারের নাম
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=1000,         # পুরো dataset একবারে load করার জন্য
    class_mode='sparse',     # integer label দিবে
    subset='training',
    shuffle=False
)

# ✅ ঠিক করা লাইন এখানে:
X, y = next(train_generator)  # ঠিক হলো এই লাইন

print("✅ X shape:", X.shape)
print("✅ y shape:", y.shape)
