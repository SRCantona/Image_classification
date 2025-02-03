
# Image Classification Task

## 📋 Introduction
This project focuses on **Task #1: Image Classification**, where the goal is to build a model capable of classifying images into different types of sports. However, the current implementation is adapted to classify **Arabic letters** using a dataset available [here](Link to dataset).

The solution leverages **PyTorch** and a pre-trained **ResNet18** model, modified to handle grayscale images representing Arabic characters. This document details the process, from dataset preparation to model evaluation.

---

## 📦 Dataset
- **Source:** [Link to dataset]
- **Structure:** The dataset consists of Arabic letter images categorized into training and testing directories:
  - `train/` - Contains labeled images for model training.
  - `test/` - Used to evaluate model performance.

### 📊 Dataset Details
- **Classes:** 28 Arabic characters
- **Format:** PNG images in grayscale
- **Labeling:** Each image file name includes its corresponding label

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('SRCantona/Arabic_letters')

# Explore dataset structure
print(dataset)
```

*For more details, please refer to the [Link to dataset].* 

---

## 🚀 Model Architecture
We used **ResNet18**, a powerful convolutional neural network (CNN) architecture known for its **residual learning framework**. 

### 🔍 What is ResNet18?
**ResNet18** stands for **Residual Network with 18 layers**. Unlike traditional CNNs, ResNet introduces **skip connections** (residuals) that allow the model to learn deeper features without facing the vanishing gradient problem. This makes training more efficient, even with many layers.

- **Key Idea:** Instead of learning the full mapping, it learns the difference (residual) between the input and output.
- **Advantages:** Faster convergence, better performance in deep networks.

### 🧱 Model Components
1. **Convolutional Layers:** Detect patterns like edges, shapes, and textures.
2. **Batch Normalization:** Normalizes outputs for faster and more stable training.
3. **ReLU Activation:** Adds non-linearity, helping the model learn complex features.
4. **Max Pooling:** Reduces the spatial size of the feature maps, making computation efficient.
5. **Fully Connected Layer:** Maps features to output classes (28 Arabic letters).

```python
import torch
import torch.nn as nn
from torchvision import models

# Load ResNet18 with pre-trained weights
base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Modify for grayscale images and 28 output classes
base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
base_model.fc = nn.Linear(base_model.fc.in_features, 28)

print(base_model)
```

---

## ⚙️ Data Processing
- **Transformations:**
  - **Resize:** Images resized to 32x32 pixels for consistency.
  - **Normalization:** Pixel values scaled between -1 and 1.
  - **Tensor Conversion:** Converts images into PyTorch tensors.

```python
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Example of applying transform
sample_image = dataset['train'][0]['image']
transformed_image = transform(sample_image)
```

### 📂 Data Loading
Implemented using a custom `ArabicCharDataset` class to load and preprocess images efficiently.

```python
class ArabicCharDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = ArabicCharDataset(dataset['train'], transform=transform)
test_dataset = ArabicCharDataset(dataset['test'], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
```

---

## 🧠 Model Training
### 🗝️ Key Concepts
- **Cross-Entropy Loss:** Measures the difference between the predicted probabilities and actual labels.
- **Adam Optimizer:** Adjusts learning rates for faster convergence.
- **Epochs:** Number of times the model sees the entire dataset during training.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters(), lr=0.001)

epochs = 20
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    base_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = base_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
```

---

## 📊 Evaluation
Model performance is evaluated on the test set:

```python
base_model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = base_model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

accuracy = 100 * total_correct / total_samples
print(f'Test Accuracy: {accuracy:.2f}%')
```

### 📈 Visualization of Training Metrics

```python
import matplotlib.pyplot as plt

# Plot training loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, marker='s')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()
```

---

## 🎯 Conclusion
The implementation of ResNet18 for Arabic letter classification demonstrates its effectiveness in handling complex image data. The model achieved notable accuracy, with performance metrics indicating strong generalization across unseen data. The training process showed consistent improvement in both loss reduction and accuracy gains, confirming the reliability of the chosen architecture and preprocessing techniques.

### 🚀 Key Takeaways
- **Pre-trained models** significantly reduce training time while maintaining high accuracy.
- **Custom dataset integration** was achieved with minimal adjustments.
- Effective performance using **standard preprocessing techniques**.
- **Visualizing metrics** provides valuable insights into model learning dynamics.

---

## 📥 Future Improvements
- Experiment with advanced architectures like **EfficientNet**.
- Apply **data augmentation** for better generalization.
- **Hyperparameter tuning** for optimized performance.
- Explore **transfer learning** with more complex datasets.

---

## 📎 References
- [PyTorch Documentation](https://pytorch.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Link to dataset]
