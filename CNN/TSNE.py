# প্রয়োজনীয় লাইব্রেরি ইমপোর্ট
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 🎯 ধরি গ্রামের ৩ ধরনের মানুষ আছে (তরুণ, মধ্যবয়সী, বৃদ্ধ)
# প্রতিজনের ১০টি বৈশিষ্ট্য (ফিচার): যেমন বয়স, আয়, জমি, গরু ইত্যাদি

# র‍্যান্ডম ডেটা তৈরি
np.random.seed(0)
group1 = np.random.normal(loc=20, scale=5, size=(100, 10))  # তরুণ
group2 = np.random.normal(loc=50, scale=5, size=(100, 10))  # মধ্যবয়সী
group3 = np.random.normal(loc=70, scale=5, size=(100, 10))  # বৃদ্ধ

# সব ডেটা একত্রে জোড়া দিচ্ছি
X = np.vstack((group1, group2, group3))  # shape: (300, 10)
y = [0]*100 + [1]*100 + [2]*100          # গ্রুপ লেবেল

# Standardization (TSNE এর জন্য ভালো Practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ TSNE দিয়ে ডেটাকে ২ ডাইমেনশনে নামানো
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_scaled)

# 🎨 Visualization (Plotting TSNE)
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', s=60)
plt.title("গ্রামের মানুষের TSNE Visualization")
plt.xlabel("TSNE Dimension 1")
plt.ylabel("TSNE Dimension 2")
plt.colorbar(label='গ্রুপ (তরুণ=0, মধ্যবয়সী=1, বৃদ্ধ=2)')
plt.grid(True)
plt.show()

# 📊 চাইলে ডেটাসেট টেবিল আকারে দেখতে পারো
df = pd.DataFrame(X_scaled, columns=[f"Feature_{i+1}" for i in range(10)])
df['Group'] = y
print(df.head())
