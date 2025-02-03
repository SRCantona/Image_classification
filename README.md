
# Image Classification - Arabic Letters

## Introduction 📚
This project focuses on building an image classification model to identify Arabic letters. Using a dataset sourced from [Hugging Face](https://huggingface.co/datasets/SRCantona/Arabic_letters), we implement a robust solution leveraging deep learning techniques with PyTorch and ResNet18 architecture.

## Dataset 📊
We utilize the Arabic Letters dataset which contains labeled images of Arabic characters. This dataset can be found [here](https://huggingface.co/datasets/SRCantona/Arabic_letters/blob/main/letter%20image.zip). It is divided into training and testing directories with each image labeled accordingly.

## Project Workflow 🚀

### 1. Device Configuration ⚡
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
This code snippet checks for GPU availability to speed up the model training process.

### 2. Arabic Alphabet Definition 📝
```python
arabic_alphabet = [' ','ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
```
Defines the set of Arabic characters that the model will classify.

### 3. Dataset Class Creation 🗂️
Handles loading images, extracting labels from filenames, and applying transformations for preprocessing.

### 4. Data Transformation and Loading 🔄
Resizes images, converts them to tensors, and normalizes for better model performance.

### 5. Model Definition 🧠
We use ResNet18, modified for grayscale images and tailored for 28 Arabic letters.

### 6. Training the Model 📈
Utilizes cross-entropy loss and the Adam optimizer to improve accuracy over 20 epochs.

### 7. Evaluation and Predictions 🎯
Visualizes predictions with Matplotlib to compare true vs predicted labels.

## Output Example 🖼️
*(Insert images showing sample predictions with correct and incorrect classifications)*

## Conclusion ✅
This project demonstrates effective image classification for Arabic letters using deep learning. Future improvements could include hyperparameter tuning and more advanced data augmentation techniques.

For more details about the dataset, visit [here](https://huggingface.co/datasets/SRCantona/Arabic_letters).
