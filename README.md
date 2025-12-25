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



**Key Observations**
- Strong performance on vehicle classes
- Confusion among visually similar animal classes
- Expected due to CIFAR-10’s low resolution (**32×32**)

---

## Visual Error Analysis

### Misclassified Test Images
<!-- Upload misclassified images here -->
<img width="970" height="397" alt="download" src="https://github.com/user-attachments/assets/39463d54-ce08-4aaa-b0b8-440910487e27" />



These examples highlight ambiguity in low-resolution images and class similarity.

---

## Real-World Smartphone Image Predictions

<img width="1189" height="755" alt="download" src="https://github.com/user-attachments/assets/bb754012-8ada-4e7f-ba18-b477f1da8c64" />




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
