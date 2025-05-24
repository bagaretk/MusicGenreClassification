# Music Genre Classification 🎵🎶

**Music genre classification using audio signal processing, convolutional neural networks, and autoencoders**

📁 _Developed by [Istrate Sebastian-Nicolae](https://github.com/bagaretk) – Politehnica University of Timișoara, 2024_

---

## 📌 Overview

This project performs automatic classification of music clips into genres using techniques from signal processing and deep learning. Starting from a baseline CNN architecture, improvements were made by adding activation functions and integrating an autoencoder to enhance the model’s performance.

🎯 **Supported genres:** `Blues`, `Classical`, `Country`, `Disco`, `Hip-hop`, `Jazz`, `Metal`, `Pop`, `Reggae`, `Rock`

---

## 🧠 Technologies Used

- Python  
- TensorFlow + Keras  
- NumPy, SciPy, Librosa  
- scikit-learn, Matplotlib  
- PyCharm + CUDA + WSL (for GPU acceleration)  
- Vast.ai (cloud GPU training)  

---

## 🗃️ Dataset

- [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html)  
- Audio clips converted into **Mel spectrograms**  
- Experiments conducted using both 2-channel and 5-channel versions of the spectrograms  

---

## 🏗️ Architecture

### Baseline Model
- 3 convolutional layers with MaxPooling and Dropout  
- Fully connected dense layer with ReLU activation  
- Softmax output layer for genre classification  

### Enhancements
- ✅ Added **ReLU activation functions** to increase model non-linearity  
- ✅ Integrated a **sparse autoencoder** for dimensionality reduction and deeper feature extraction  

---

## 📊 Results

The improved architecture significantly outperforms the baseline:

- Higher accuracy with ReLU activations  
- Better performance with richer (5-channel) spectrograms  

---

## 🚀 How to Run the Project

```bash
git clone https://github.com/bagaretk/MusicGenreClassification.git
cd MusicGenreClassification

python -m venv venv
source venv/bin/activate    # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Preprocessing
python preprocess.py

# Postprocessing
python postprocess.py

# CNN training
python train_cnn.py

# Autoencoder training
python train_autoencoder.py
```
## 📁 Project Structure
--- 
- ├── mel-spec
- ├──├── code # where all the magic happens

---

## 📈 Future Work

- Create a mobile app for real-time music genre prediction  
- Expand to larger and more diverse datasets  
- Experiment with different Autoencoder architectures  

---

## 📚 References

- [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)  
- [TensorFlow](https://www.tensorflow.org/)  
- [Librosa](https://librosa.org/)  

