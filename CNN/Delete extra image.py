import os
import shutil

# মূল dataset ফোল্ডার
dataset_dir = "dataset"

# প্রতি ক্লাসে রাখতে চাওয়া ইমেজ সংখ্যা
keep_per_class = 100

# প্রতিটি ক্লাস ফোল্ডার ধরে কাজ করা
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)

    if not os.path.isdir(class_path):
        continue  # যদি ফোল্ডার না হয়, স্কিপ কর

    images = sorted(os.listdir(class_path))  # ফাইলগুলো alphabetically sort
    total_images = len(images)

    if total_images <= keep_per_class:
        print(f"✅ Skipping {class_name}, only {total_images} images found.")
        continue

    # যেগুলো delete হবে তাদের লিস্ট
    images_to_delete = images[keep_per_class:]

    # Delete images
    for image_file in images_to_delete:
        file_path = os.path.join(class_path, image_file)
        os.remove(file_path)

    print(f"🗑️ Deleted {len(images_to_delete)} images from class '{class_name}'")

print("✅ Cleanup complete: Only 100 images kept per class.")
