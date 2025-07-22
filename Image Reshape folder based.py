from PIL import Image
import os

# 🔧 রিসাইজ করার জন্য নতুন সাইজ (আপনি ইচ্ছেমতো বদলাতে পারেন)
new_size = (28, 28)

# 📂 আপনার ফোল্ডারের পাথ
folder_path = r"C:\Users\uiu\Desktop\TEST"

# 📁 ফোল্ডারে থাকা সব ফাইল চেক করা
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        image_path = os.path.join(folder_path, filename)
        try:
            with Image.open(image_path) as img:
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                resized_img.save(image_path)
                print(f"✅ {filename} রিসাইজ সম্পন্ন হয়েছে")
        except Exception as e:
            print(f"❌ {filename} প্রক্রিয়াজাত করতে সমস্যা হয়েছে: {e}")
