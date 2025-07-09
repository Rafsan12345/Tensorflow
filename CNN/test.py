import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ১. আগে আপনার .h5 মডেল ফাইল লোড করুন
model = load_model('cnn_model_digits_0_to_9.h5')  # আপনার সেভ করা মডেল ফাইলের নাম দিন

# ২. টেস্ট করার জন্য ছবি লোড করুন (28x28, গ্রেস্কেল)
img_path = "C:\\Users\\uiu\\Desktop\\Tensorflow\\dataset\\0\\10000.png"  # এখানে আপনার ছবি ফাইলের পাথ দিন

img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')

# ৩. ছবিকে numpy array তে রূপান্তর করুন
img_array = image.img_to_array(img)

# ৪. ডাইমেনশন বাড়ান (batch size এর জন্য) এবং normalize করুন
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)  # shape হবে (1, 28, 28, 1)

# ৫. মডেল দিয়ে প্রেডিক্ট করুন
pred = model.predict(img_array)

# ৬. প্রেডিক্টেড ক্লাস বের করুন
predicted_class = np.argmax(pred, axis=1)[0]
print(f"Predicted class: {predicted_class}")
