
# 🐾   Animal Sound Classification Using Spectrograms with ML
   ### 🎧 Deep Learning with TensorFlow + MobileNetV2 + MongoDB

This project classifies **animal sounds (dog vs cat)** using spectrogram images generated from audio files.
Spectrograms are created using **Librosa**, and a deep learning model is trained using **MobileNetV2**.
Prediction results are stored in **MongoDB**.

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
stft = librosa.stft(signal, n_fft=2048, hop_length=512, win_length=2048) <br>
spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

# ✔ Output Sample
<img src="img/screenshot.png" width="700">


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
<br>
Accuracy & Validation Accuracy
<p align="center"> <img src="img/accuracy_graph.png" width="700"> </p> <br>
Loss & Validation Loss
<p align="center"> <img src="img/loss_graph.png" width="700"> </p>


