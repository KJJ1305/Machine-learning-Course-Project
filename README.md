

# **README.md**

# 🩸 Blood Cell Classification Using EfficientNet-B4

*A deep learning project for classifying white blood cells with high accuracy*

---

## 📌 **Project Overview**

This project uses **EfficientNet-B4**, a modern convolutional neural network pretrained on ImageNet, to classify microscopic blood cell images into four classes:

* **Eosinophil**
* **Lymphocyte**
* **Monocyte**
* **Neutrophil**

The model is fine-tuned using heavy data augmentation, label smoothing, class balancing, and Test-Time Augmentation (TTA), achieving **~89% overall accuracy** on the test set.

---

## 📂 **Dataset Structure**

Your dataset should follow the ImageFolder format:

```
images/
│
├── TRAIN/
│   ├── EOSINOPHIL/
│   ├── LYMPHOCYTE/
│   ├── MONOCYTE/
│   └── NEUTROPHIL/
│
└── TEST/
    ├── EOSINOPHIL/
    ├── LYMPHOCYTE/
    ├── MONOCYTE/
    └── NEUTROPHIL/
```

---

## 🚀 **Features**

✔ EfficientNet-B4 pretrained on ImageNet
✔ Heavy training augmentations (crop, rotate, distort, erase)
✔ Label Smoothing Cross-Entropy
✔ Class weights to boost minority class performance
✔ Mixed precision training (AMP)
✔ Cosine Annealing LR Scheduler
✔ Test-Time Augmentation (TTA)
✔ Confusion matrix + classification report

---

## 🧠 **Model Architecture**

* **Backbone:** EfficientNet-B4 (19M parameters)
* **Custom classifier head:**

  ```
  Dropout(0.4)
  Linear(1792 → 512)
  ReLU
  Dropout(0.3)
  Linear(512 → 4)
  ```

All layers are fine-tuned.

---

## 🛠️ **Installation**

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install timm
pip install scikit-learn matplotlib tqdm seaborn
```

*(Colab users already have most packages.)*

---

## ▶️ **Training**

Run the training script:

```python
python train.py
```

The script will:

* Load dataset
* Apply augmentations
* Train EfficientNet-B4
* Save the best model as:

```
best_efficientnet_b4.pth
```

---

## 🧪 **Testing (With TTA)**

To run only the test phase:

```python
python test.py --weights best_efficientnet_b4.pth
```

This performs:

* Model loading
* TTA inference
* Classification report
* Confusion matrix

---

## 📊 **Final Test Results**

### **Classification Report**

```
              precision    recall   f1-score   support
EOSINOPHIL      0.9354     0.8828     0.9083     623
LYMPHOCYTE      1.0000     1.0000     1.0000     620
MONOCYTE        1.0000     0.7500     0.8571     620
NEUTROPHIL      0.7199     0.9391     0.8150     624

Overall Accuracy: 0.8930
```

---

## 📉 **Confusion Matrix**

```
               Predicted
               E    L    M    N
True
Eosinophil    550   0    0   73
Lymphocyte      0  620   0    0
Monocyte        0   0   465  155
Neutrophil     38   0    0   586
```

---

## 🧾 **Project Files**

```
├── train.py            # Full training pipeline
├── test.py             # Test-only script with TTA
├── utils.py            # Helper functions (optional)
├── README.md           # Project documentation
└── best_efficientnet_b4.pth  # Saved model weights
```

---

## ⭐ **If you use this project, please star the repo!**

It helps support further development.

---


