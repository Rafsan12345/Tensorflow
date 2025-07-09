import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Dataset directory path
DATASET_DIR = "dataset"  # আপনার dataset ফোল্ডারের পাথ দিন

# ডেটা লোড করুন (grayscale, batch size বড় রাখুন যাতে সব ডেটা একসাথে পাওয়া যায়)
datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=1000,  # বড় ব্যাচ সাইজ
    shuffle=False
)

# সব ডেটা একসাথে নিন
X, y = next(generator)

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")

# NaN অথবা inf চেক করুন
if np.isnan(X).any() or np.isinf(X).any():
    print("🚫 Data contains NaNs or infinite values!")
    # NaN/inf থাকলে সরান অথবা ফিক্স করুন
    X = np.nan_to_num(X)

# সম্পূর্ণ zeros চেক (সব পিক্সেল zero থাকলে)
zero_images = np.all(X == 0, axis=(1,2,3))
if np.any(zero_images):
    print(f"⚠️ Found {np.sum(zero_images)} zero images. They will be removed.")
    X = X[~zero_images]
    y = y[~zero_images]

# 4D to 2D reshape (for PCA/TSNE)
X_flat = X.reshape(X.shape[0], -1)

# PCA দিয়ে dimension কমান (TSNE আগে)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_flat)

# TSNE প্রয়োগ করুন (এখন 50D থেকে 2D)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# প্লটিং
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10)
plt.colorbar(scatter, ticks=range(10))
plt.title("TSNE visualization of dataset")
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.show()
