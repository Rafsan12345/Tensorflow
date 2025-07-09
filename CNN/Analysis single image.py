from PIL import Image
import numpy as np

# ছবি লোডের পাথ
img_path = r"C:\Users\uiu\Desktop\CNN\dataset\9\0.png"

# ছবি ওপেন করুন
img = Image.open(img_path)

# numpy array এ রূপান্তর করুন
img_arr = np.array(img)

# তথ্য সংগ্রহ
mode = img.mode
shape = img_arr.shape
min_pixel = np.min(img_arr)
max_pixel = np.max(img_arr)
mean_pixel = np.mean(img_arr)
std_pixel = np.std(img_arr)
is_black = np.all(img_arr == 0)

# প্রিন্ট করুন
print(f"Image mode: {mode}")
print(f"Image shape: {shape}")
print(f"Pixel min value: {min_pixel}")
print(f"Pixel max value: {max_pixel}")
print(f"Pixel mean value: {mean_pixel:.4f}")
print(f"Pixel std deviation: {std_pixel:.4f}")
print(f"Is completely black? {is_black}")
