# 🖼️ AI-Detector: Fingerprinting AI-Generated Images

**Author:** Gianluca Giuseppe Maria De Nardi  
**Type:** Technical Report  
**Topic:** AI-generated image detection through entropy-based fingerprinting

---

## 📌 Overview
AI-Detector is a project designed to identify **AI-generated images** by extracting their **fingerprints**.  
Instead of relying on traditional noise-based methods, this approach uses **entropy differences** between **low-texture** and **high-texture** regions, combined with **high-pass filtering** and **deep neural networks**.

The method focuses on highlighting artifacts such as:
- Uneven colors
- Repetitive patterns
- Black patches
- Inconsistent noise

---

## ⚙️ Methodology

### 1. Patch Extraction
- Images are divided into **8×8 patches**.
- **64 rich-texture** patches and **64 low-texture** patches are selected.

### 2. Fingerprint Generation
- Patches are recomposed into two images (rich & poor texture).
- A set of **30 high-pass filters (5×5 kernels)** is applied.
- High-frequency details are extracted to emphasize visual inconsistencies.

### 3. Classification
- Fingerprint images are passed through different classifiers:
  - **CNN (4 convolutional layers)**
  - **ResNet50 (50 layers)**
  - **ResNet101 (101 layers)**
  - **ResNet151 (152 layers)**

---

## 📂 Datasets

Two Kaggle datasets were used via API:

1. **AI Generated Images vs Real Images**
   - `AiArtData` (label = 0) → AI-generated  
   - `RealArt` (label = 1) → Real  
   - 973 images (538 fake, 435 real)

2. **Camera Photos vs AI Generated Photos**
   - `Ai_Images` (label = 0)  
   - `Camera_images` (label = 1)  
   - ~300 high-resolution images per class

📌 Data augmentation was applied to compensate for limited dataset size.  

---

## 🔧 Training Setup
- **Loss:** `nn.BCELoss()` (Binary Cross-Entropy)  
- **Optimizer:** Adam (adaptive learning rate, bias correction)  
- **Scheduler:** Learning rate decay over time  
- **Input size:** `[3, 128, 128]`  
- **Output:** `[3, 128, 128]` high-frequency fingerprint map  

---

## 📊 Results

| Classifier  | Accuracy | Loss | Epochs |
|-------------|----------|------|--------|
| CNN         | 59.28%   | 0.53 | 600    |
| ResNet50    | 70.33%   | 0.65 | 70     |
| **ResNet101** | **79%**   | 0.52 | 100    |
| ResNet151   | 73.63%   | 0.56 | 100    |

**Confusion Matrix Metrics (ResNet101):**
- **Precision:** AI = 0.68 | Camera = 0.77  
- **Recall:** AI = 0.76 | Camera = 0.69  
- **F1-score:** AI = 0.72 | Camera = 0.73  

📌 **Best performing model:** ResNet101 (~79% accuracy)

---

## 🚀 Future Work
- Introduce **Swin Transformer** with sliding window attention before high-pass filtering  
- Improve dataset quality and diversity  
- Explore **regularization techniques** to stabilize validation loss  
- Run experiments on **high-performance GPUs** to scale training  

---

## 📚 References
1. Durall et al., *Watch your up-convolution*, CVPR 2020  
2. Frank et al., *Frequency analysis for deep fake recognition*, ICML 2020  
3. [AI-Generated Image Detection GitHub](https://github.com/dustin-nguyen-qil/AI-Generated-Image-Detection)  
4. Hmidi & Jihene, *Deep residual network in network*, 2021  
5. Susladkar et al., *ClarifyNet*, 2022  

---

🔹 This project provides a **fingerprint-based deep learning pipeline** to detect AI-generated images and paves the way for **more robust detectors** using modern transformer-based architectures.
