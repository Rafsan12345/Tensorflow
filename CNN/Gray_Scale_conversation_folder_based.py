from PIL import Image
import os

# ✅ ইনপুট ফোল্ডার
input_folder = r"C:\Users\uiu\Downloads\grAy" # নিজের path দিন

# ✅ আউটপুট ফোল্ডার (একই ফোল্ডারে new নাম)
output_folder = os.path.join(input_folder, "grayscale")
os.makedirs(output_folder, exist_ok=True)

# ✅ সব ইমেজ প্রসেস করুন
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('L')  # 'L' = grayscale

        # নতুন ফাইল নাম: originalname_gray.png
        base_name, ext = os.path.splitext(filename)
        new_filename = base_name + "_gray.png"
        output_path = os.path.join(output_folder, new_filename)

        # সেভ করুন
        img.save(output_path)

        print(f"✅ Saved: {new_filename}")

print("🎉 All RGB images converted to grayscale and saved.")
