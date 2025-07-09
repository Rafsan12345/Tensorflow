from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 1️⃣ মডেল লোড
model = load_model("cnn_model_rgba.h5")  # আপনার মডেল ফাইল নাম দিন

# 2️⃣ ছবি লোড
img_path = r"C:\Users\uiu\Desktop\CNN\dataset\9\10002.png"
img = image.load_img(img_path, target_size=(28, 28), color_mode='rgba')  # RGBA

# 3️⃣ প্রিপ্রসেসিং
img_array = image.img_to_array(img) / 255.0  # normalize
img_array = img_array.reshape(1, 28, 28, 4)   # 4 চ্যানেল

# 4️⃣ প্রেডিকশন
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print("✅ Predicted class:", predicted_class)

