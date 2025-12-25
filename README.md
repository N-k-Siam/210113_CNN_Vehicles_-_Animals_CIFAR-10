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
│ ├── .gitkeep<br>
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

### Training Curves

#### Loss vs Epoch
<img width="567" height="455" alt="download" src="https://github.com/user-attachments/assets/0d771ff0-2b12-40fd-a2f2-d563c12408d0" />



#### Accuracy vs Epoch
<img width="576" height="455" alt="download" src="https://github.com/user-attachments/assets/af57ef38-5702-4c08-a52a-4ed6eac3a9c2" />


### Observations
- Training loss decreases steadily
- Training accuracy improves consistently
- Validation accuracy stabilizes around **75–76%**
- Overfitting begins after later epochs (expected behavior)


---

## Evaluation on CIFAR-10 Test Set

### Confusion Matrix
<img width="853" height="766" alt="download" src="https://github.com/user-attachments/assets/cbe68edd-c2d2-494b-bac5-3648d527b978" />


**Key Observations**
- Strong performance on vehicle classes
- Confusion among visually similar animal classes
- Expected due to CIFAR-10’s low resolution (**32×32**)

---

## Visual Error Analysis

### Misclassified Test Images
<!-- Upload misclassified images here -->
<img width="970" height="397" alt="download" src="https://github.com/user-attachments/assets/89c26d57-aa72-47f7-83ab-14f879526627" />


These examples highlight ambiguity in low-resolution images and class similarity.

---

## Real-World Smartphone Image Predictions


<img width="1189" height="707" alt="download" src="https://github.com/user-attachments/assets/cf947b9e-6226-4ec8-b9fb-79c07053b12f" />


**Observations**
- Vehicle images are classified with high confidence
- Animal classes occasionally show confusion
- Confidence varies due to domain shift between CIFAR-10 and real-world images

---

## How to Run (Google Colab)

1. Open the notebook in Google Colab:  
   Click here to open [`210113.ipynb`](https://colab.research.google.com/drive/1H0BBNrzxFN-SPrRPr2kSWs63XaEaWGUg?usp=sharing) in Colab

2. Select **Runtime → Run all**

All training, evaluation, and visualizations will be executed automatically.  
No manual cloning or file uploads are required.


**Author**<br>
**Nafiullah Khan Siam**<br>
**Student ID: 210113**
