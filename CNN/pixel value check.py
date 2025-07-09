from PIL import Image
import numpy as np

img_path = r"C:\Users\uiu\Desktop\CNN\dataset\0\1000.png"
img = Image.open(img_path).convert('L')  # grayscale এ রূপান্তর
img_arr = np.array(img) / 255.0  # normalize

print("Image shape:", img_arr.shape)
print("Pixel min value:", img_arr.min())
print("Pixel max value:", img_arr.max())
print("Is completely black?", np.all(img_arr == 0))

# numpy print option সেট করুন যাতে পুরো array প্রিন্ট হয়
np.set_printoptions(threshold=np.inf)

# 1D আকারে পিক্সেল ভ্যালু দেখান
print("Image pixel values (1D):")
print(img_arr.flatten())

