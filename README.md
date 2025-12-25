# Vehicles & Animals Image Classification with CIFAR-10 (CNN)

This project presents a complete **Convolutional Neural Network (CNN)** based image classification pipeline implemented using **PyTorch**.  
The model is trained on the **CIFAR-10 dataset** and evaluated on both:
- the standard CIFAR-10 test set, and  
- **real-world smartphone images** to analyze generalization performance.

The notebook is fully automated and can be executed using **Run All** in Google Colab.

---

## Dataset

### Standard Dataset
- **CIFAR-10** (10 classes):
  - airplane, automobile, bird, cat, deer  
  - dog, frog, horse, ship, truck

### Custom Dataset (Real-world Images)
- Real-world images captured using a smartphone
- Used to evaluate model performance on unseen real-world data

---

## Project Structure

<!-- Upload project structure image here -->
210113_CNN_Vehicles_-_Animals_CIFAR-10/<br>
├── dataset/ <br>
├── model/<br>
│ └── 210113.pth <br>
├── 210113.ipynb <br>
└── README.md<br>

---

## Model Architecture
- CNN implemented **from scratch** using `torch.nn.Module`
- Key components:
  - Convolution + ReLU layers
  - MaxPooling layers
  - Fully Connected layers
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam

---

## Training
- Dataset automatically downloaded using `torchvision.datasets`
- Images preprocessed using `torchvision.transforms`
- Train/Validation split applied
- Training and validation performance tracked across epochs

---

## Training & Validation Results

### Training Log (Per Epoch)
Epoch 1/10 | Train loss=1.4306, acc=0.4792 | Val loss=1.1085, acc=0.5956<br>
Epoch 2/10 | Train loss=1.0077, acc=0.6428 | Val loss=0.8827, acc=0.6870<br>
Epoch 3/10 | Train loss=0.8386, acc=0.7045 | Val loss=0.8156, acc=0.7148<br>
Epoch 4/10 | Train loss=0.7162, acc=0.7489 | Val loss=0.7539, acc=0.7350<br>
Epoch 5/10 | Train loss=0.6225, acc=0.7805 | Val loss=0.7380, acc=0.7418<br>
Epoch 6/10 | Train loss=0.5576, acc=0.8042 | Val loss=0.7646, acc=0.7430<br>
Epoch 7/10 | Train loss=0.4828, acc=0.8289 | Val loss=0.7666, acc=0.7486<br>
Epoch 8/10 | Train loss=0.4248, acc=0.8466 | Val loss=0.7717, acc=0.7542<br>
Epoch 9/10 | Train loss=0.3719, acc=0.8661 | Val loss=0.8016, acc=0.7432<br>
Epoch 10/10 | Train loss=0.3329, acc=0.8799 | Val loss=0.7982, acc=0.7548<br>


### Observations
- Training loss decreases steadily across epochs
- Training accuracy increases consistently, reaching ~88%
- Validation accuracy stabilizes around **75–76%**
- Validation loss increases in later epochs, indicating overfitting
- Best generalization performance is observed around Epoch 7–8

---


### Training Curves

#### Loss vs Epoch
<img width="567" height="455" alt="download" src="https://github.com/user-attachments/assets/ec1bb977-5139-4471-ac7b-abf41897ed5b" />




#### Accuracy vs Epoch
<img width="576" height="455" alt="download" src="https://github.com/user-attachments/assets/faf9d9be-4f5d-4689-9bed-927a3fd6b817" />



### Observations
- Training loss decreases steadily
- Training accuracy improves consistently
- Validation accuracy stabilizes around **75–76%**
- Overfitting begins after later epochs (expected behavior)


---

## Evaluation on CIFAR-10 Test Set

### Confusion Matrix
<img width="853" height="766" alt="download" src="https://github.com/user-attachments/assets/4086501d-accd-4c7f-ab71-9d5c13681613" />



### Key Observations
- Strong performance on vehicle classes
- Confusion among visually similar animal classes
- Expected due to CIFAR-10’s low resolution (**32×32**)

---
## Real-World Smartphone Image Predictions
<img width="950" height="665" alt="download" src="https://github.com/user-attachments/assets/c3f0448c-194f-496b-afb6-470a513408e7" />






### Observations
- Vehicle images are classified with high confidence
- Animal classes occasionally show confusion
- Confidence varies due to domain shift between CIFAR-10 and real-world images

---
## Visual Error Analysis

### Misclassified Test Images
<!-- Upload misclassified images here -->

<img width="970" height="397" alt="download" src="https://github.com/user-attachments/assets/19dd1337-e107-4797-b8ac-f93e0e3d32ba" />



These examples highlight ambiguity in low-resolution images and class similarity.

---

## How to Run (Google Colab)

1. Open the notebook in Google Colab:  
   Click here to open [`210113.ipynb`](https://colab.research.google.com/drive/1H0BBNrzxFN-SPrRPr2kSWs63XaEaWGUg?usp=sharing) in Colab

2. Select **Runtime → Run all**

All training, evaluation, and visualizations will be executed automatically.  
No manual cloning or file uploads are required.


# **Author**<br>
# **Nafiullah Khan Siam**<br>
# **Student ID: 210113**
