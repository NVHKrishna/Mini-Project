import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2

from pymongo import MongoClient
from datetime import datetime


# -------------------- DATA LOADING --------------------
data_train_path = r"C:\Users\NVH Krishna\Desktop\mini_project\dog_cat_data_specs_color_split\train"
data_valid_path = r"C:\Users\NVH Krishna\Desktop\mini_project\dog_cat_data_specs_color_split\val"
data_test_path  = r"C:\Users\NVH Krishna\Desktop\mini_project\dog_cat_data_specs_color_split\test"


# Load raw dataSets
train_raw = tf.keras.utils.image_dataset_from_directory(
    data_train_path, shuffle=False, image_size=(224, 224), batch_size=32
)
valid_raw = tf.keras.utils.image_dataset_from_directory(
    data_valid_path, shuffle=False, image_size=(224, 224), batch_size=32
)
test_raw = tf.keras.utils.image_dataset_from_directory(
    data_test_path, shuffle=False, image_size=(224, 224), batch_size=32
)

# Save class names before mapping (since .map removes them)
class_names = train_raw.class_names
num_classes = len(class_names)
print(" Classes found:", class_names) 


# Normalize pixel values for better training
AUTOTUNE = tf.data.AUTOTUNE
data_train = train_raw.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(AUTOTUNE)
data_valid = valid_raw.map(lambda x, y: (x / 255.0, y)).cache().prefetch(AUTOTUNE)
data_test  = test_raw.map(lambda x, y: (x / 255.0, y)).cache().prefetch(AUTOTUNE)

# --------------------  CNN MODEL BUILDING --------------------
model_path = "animal_sound_mobilenetv2.h5"

if not os.path.exists(model_path):
    print("Training MobileNetV2 model for the First Time.....")

#  // Pre-Trained model that is Mobilenetv2 
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  #freeze pretrained weights

#  // building top layer of the base model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Training
    epochs_size = 25
    history = model.fit(data_train, validation_data=data_valid, epochs=epochs_size)

    # Save model
    model.save(model_path)
    print(f" Model trained and saved as {model_path}")

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

else:
    print(" Model already exists — loading saved model.")
    model = tf.keras.models.load_model(model_path)


# -------------------- TEST / PREDICTION -----------------------------------------------------------
image_path = r"C:\Users\NVH Krishna\Desktop\mini_project\img\dog0021_spectrogram.png"
img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_batch = tf.expand_dims(img_array, 0)

predictions = model.predict(img_batch)
score = tf.nn.softmax(predictions[0])

predicted_label = class_names[np.argmax(score)]
confidence = np.max(score) * 100

print(f"\n🎯 Predicted: {predicted_label} with confidence {confidence:.2f}%")   # FINAL-output 


#  Display all class confidences
print("\n Confidence levels for each class:")
for i, class_name in enumerate(class_names):
    print(f"   {class_name:15s} — {score[i]*100:.2f}%")

# Sorted confidence display
sorted_indices = np.argsort(score)[::-1]
print("\n Confidence (sorted):")
for i in sorted_indices:
    print(f"   {class_names[i]:15s} — {score[i]*100:.2f}%")




# -------------------- MONGODB INTEGRATION --------------------
print("\n📦 Connecting to MongoDB.....")

try:
    # Local MongoDB 
    MONGO_URI = "mongodb://localhost:27017/"
    client = MongoClient(MONGO_URI)

    db = client["Animal_Voice_Model"]
    collection = db["Predictions"]
    
    print("✅ Connected to MongoDB !!")

    # Round confidence to 2 decimal places
    confidence = round(float(confidence), 2)

    # Prepare record
    record = {
        "file_name": os.path.basename(image_path),
        "predicted_class": predicted_label,
        "confidence": confidence,  # only 2 decimals now
        "all_class_confidences": {
            class_names[i]: float(score[i]) for i in range(len(class_names))
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Insert into MongoDB
    collection.insert_one(record)
    print("💾 Prediction result saved successfully to MongoDB! \n")

except Exception as e:
    print("❌ MongoDB save failed:", e)
