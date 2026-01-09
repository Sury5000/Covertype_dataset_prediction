# PyTorch Classification – Forest Cover Type Prediction

This project documents my end-to-end implementation of a **multi-class classification pipeline using PyTorch**, applied to the **Forest Cover Type dataset**. The work focuses on building custom neural network components, handling large tabular data, training with proper validation, and performing hyperparameter optimization.

---

## Project File

- pytorch_classification.py – Complete PyTorch classification pipeline

---

## Objective

The objective of this work is to:
- Implement a classification model using PyTorch from low-level to high-level abstractions
- Handle a large real-world tabular dataset
- Design a regularized neural network for multi-class classification
- Optimize model performance using validation and hyperparameter tuning

---

## Dataset – Forest Cover Type

The dataset used is **Forest Cover Type**, fetched using `sklearn.datasets.fetch_covtype`.

- Features: 54 numerical features
- Target: 7 forest cover type classes
- Dataset size: over 500,000 samples
- Target labels adjusted to start from 0 (PyTorch requirement)

The data is split into training, validation, and test sets with stratification.

---

## Data Preprocessing

Steps performed:
- Standardized features using `StandardScaler`
- Converted NumPy arrays into PyTorch tensors
- Implemented a custom `Dataset` class (`CoverTypeDataset`)
- Created efficient `DataLoader`s with batching, shuffling, and multiprocessing

This ensures scalable and efficient model training.

---

## Custom Neural Network Components

### High-Level Dense Layer
- Implemented using `nn.Linear` and `nn.ReLU`
- Used to inspect parameters and understand PyTorch abstractions

### Low-Level Dense Layer
- Implemented manually using `nn.Parameter`
- Used explicit matrix multiplication and bias addition
- Applied Kaiming initialization
- Demonstrates low-level control over neural network operations

---

## Final Model Architecture

A fully connected neural network (`CoverTypeClassifier`) was built with:
- Custom dense blocks (Linear → BatchNorm → ReLU)
- Dropout layers for regularization
- Two hidden layers with configurable dimensions
- Output layer producing logits for 7 classes

The model is designed for tabular data classification.

---

## Training and Evaluation Pipeline

Reusable training and evaluation functions were implemented:
- Training loop with gradient backpropagation
- Accuracy computation per epoch
- Separate evaluation loop using `torch.inference_mode()`
- GPU acceleration when available

This provides a clean and controlled training workflow.

---

## Hyperparameter Optimization

Hyperparameter tuning was performed using **Optuna**, optimizing:
- Hidden layer dimension
- Learning rate
- Dropout probability
- Optimizer choice (AdamW / RMSprop)

Validation accuracy was used as the optimization metric, with pruning applied for efficiency.

---

## Final Training Strategy

- Best hyperparameters selected from Optuna study
- Model retrained using learning-rate scheduling (`ReduceLROnPlateau`)
- Best model checkpoint saved based on validation accuracy
- Final evaluation performed on the test dataset

Additionally, the model was retrained on the **full training dataset** to maximize performance.

---

## Results

- Achieved strong validation and test accuracy on a large-scale dataset
- Regularization (BatchNorm + Dropout) improved stability
- Learning-rate scheduling helped converge to better performance

---

## Conclusion

This project demonstrates a complete PyTorch classification workflow on real-world tabular data. By combining custom neural network components, efficient data handling, hyperparameter optimization, and structured training logic, the work reflects a solid understanding of practical deep learning using PyTorch.

---
