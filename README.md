
# 🐾   Animal Sound Classification Using Spectrograms with ML <br>
   ### 🎧 Deep Learning with TensorFlow + MobileNetV2 + MongoDB

This project presents a complete pipeline for **Animal sound classification (Cat vs Dog)** using Deep learning and Audio signal processing **(STFT)**. Instead of relying on raw audio, the system transforms sound waves into visually interpretable **spectrogram images**, enabling powerful image-based models like **MobileNetV2** to learn frequency patterns effectively. The spectrograms are generated from real-world audio samples—processed, organized, and fed into a **TensorFlow-powered classifier**. Finally, each prediction is logged into **MongoDB**,   creating a robust audio-analysis workflow suitable for Machine learning projects, Academic research, or Real-time applications.

---

# ✨ Overview

1. Convert audio (WAV) → Spectrogram (PNG)
2. Using Windsurf (automation assistant) to Generate Spectrogram Datasets
3. Build training dataset from spectrogram images generated from Windsurf 
4. Train MobileNetV2 classifier using TensorFlow with Keras
5. Predict new Animal sounds  
6. Save prediction results to MongoDB Database

---

# 🛠 Technologies Used

- Python 3.x
- Windsurf(automation assistant)
- TensorFlow / Keras  
- MobileNetV2  
- Librosa  
- Matplotlib  
- SoundFile  
- MongoDB  
- pymongo  

---

# 📂 Project Structure
<img src="Screenshot 2025-11-14 195047.png" width="700">


<br>

# 📊 Spectrogram Generation

<img src="Screenshot 2025-11-14 194559.png" width="700">


# ✔ Code Used
  <img src="Screenshot 2025-11-14 204237.png" width="700">


# ✔ Output Sample
<img src="Screenshot 2025-09-11 111118.png" width="800">

# 🎙 Dataset Description

The dataset consists of audio clips of:

### -- Cat sounds datasets

### -- Dog sounds datasets

Dataset source: Kaggle
Example:
https://www.kaggle.com

  <img src="Screenshot 2025-11-14 205008.png" width="700">

<br>

# 🧠 Model Architecture (MobileNetV2)

The classifier uses Transfer Learning:

  ### <u>Base Model:</u>


-- MobileNetV2(include_top=False)  <br>
-- Pretrained on ImageNet <br>
-- Frozen weights (for feature extraction) <br>

<br>

 ### <u>Advantages: </u>


✔ Faster training  <br>
✔ High accuracy even with small datasets  <br>
✔ Lightweight and efficient <br>

# 🏋️ Training Process
### <u>Steps :</u>


1. Load datasets from directory <br>
2. Normalize pixel values <br>
3. Cache + prefetch for fast pipeline <br>
4. Train MobileNetV2 for 25 epochs <br>
5. Save model as animal_sound_mobilenetv2.h5 <br>
   


 #  🎯 Making Predictions
<img src="Screenshot 2025-11-14 200316.png" width="700">



 # 🗄 MongoDB Logging
<img src="Screenshot 2025-11-14 200517.png" width="700" >


# 📈 Training Graphs 
## <u>Accuracy & Validation Accuracy</u>

<p align="center"> <img src="Screenshot 2025-11-14 203627.png" width="600"> </p> <br>

## <u>Loss & Validation Loss</u>
<p align="center"> <img src="Screenshot 2025-11-14 203649.png" width="600"> </p>

                                 # 🎉 Done!

