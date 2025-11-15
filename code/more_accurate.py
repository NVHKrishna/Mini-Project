import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# -------------------- DATA LOADING --------------------
data_train_path = r"C:\Users\NVH Krishna\Desktop\mini_project\training dataset_animals\train"
data_test_path  = r"C:\Users\NVH Krishna\Desktop\mini_project\training dataset_animals\test"
data_valid_path = r"C:\Users\NVH Krishna\Desktop\mini_project\training dataset_animals\valid"

data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path, shuffle=True, image_size=(224, 224), batch_size=32
)
data_valid = tf.keras.utils.image_dataset_from_directory(
    data_valid_path, shuffle=True, image_size=(224, 224), batch_size=32
)
data_test = tf.keras.utils.image_dataset_from_directory(
    data_test_path, shuffle=False, image_size=(224, 224), batch_size=32
)

# ✅ Save class names before dataset mapping
class_names = data_train.class_names
num_classes = len(class_names)

# -------------------- DATA NORMALIZATION --------------------
def normalize_images(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

data_train = data_train.map(normalize_images)
data_valid = data_valid.map(normalize_images)
data_test  = data_test.map(normalize_images)

# -------------------- MODEL BUILDING --------------------
model_path = "animal_sound_model.h5"

if not os.path.exists(model_path):
    print("🚀 Training model for the first time...")

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # Train model
    epochs_size = 30
    history = model.fit(data_train, validation_data=data_valid, epochs=epochs_size)

    # Save model
    model.save(model_path)
    print(f"✅ Model trained and saved as {model_path}")

    # -------------------- PLOT TRAINING HISTORY --------------------
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
    print("✅ Model already exists — skipping training.")
    model = tf.keras.models.load_model(model_path)

# -------------------- EVALUATE MODEL --------------------
test_loss, test_acc = model.evaluate(data_test)
print(f"\n✅ Test accuracy: {test_acc * 100:.2f}%")

# -------------------- TEST / PREDICTION --------------------
image_path = r"C:\Users\NVH Krishna\Desktop\mini_project\training dataset_animals\train\Cow\Cow_16_spec_png.rf.52b2a9a0de7f1811a05b038a719df4cc.jpg"
img = tf.keras.utils.load_img(image_path, target_size=(224,224))
img_array = tf.keras.utils.img_to_array(img)
img_batch = tf.expand_dims(img_array, 0)

# Predict
predictions = model.predict(img_batch)
score = tf.nn.softmax(predictions[0])

predicted_label = class_names[np.argmax(score)]
confidence = np.max(score) * 100

print(f"\n🎯 Predicted: {predicted_label} with confidence {confidence:.2f}%")

# 🔍 Display confidence for all classes
print("\n📊 Confidence levels for each class:")
for i, class_name in enumerate(class_names):
    print(f"   {class_name:15s} — {score[i]*100:.2f}%")

# (Optional) Sorted confidence display
sorted_indices = np.argsort(score)[::-1]
print("\n📈 Confidence (sorted):")
for i in sorted_indices:
    print(f"   {class_names[i]:15s} — {score[i]*100:.2f}%")
