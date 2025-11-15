import os

base_path = r"C:\Users\NVH Krishna\Desktop\mini_project\output_spectrograms_226"

for folder in ["train", "val", "test"]:
    folder_path = os.path.join(base_path, folder)
    print(f"\n🔍 Checking: {folder_path}")
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  {subdir:10s} → {len(files)} image files")

import tensorflow as tf

data_train_path = r"C:\Users\NVH Krishna\Desktop\mini_project\output_spectrograms_226\train"

train_raw = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(224, 224),
    batch_size=32
)

print("\n✅ Loaded", train_raw.cardinality().numpy() * 32, "images (approx.)")
