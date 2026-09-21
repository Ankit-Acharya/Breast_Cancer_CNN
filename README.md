# 🧬 Breast Cancer Histopathology Classification Using CNN

<p align="center">
  <strong>Binary Classification of Breast Histopathology Images with TensorFlow and Keras</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Computer%20Vision-CNN-5C3EE8?style=for-the-badge" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black" alt="Google Colab">
</p>

<p align="center">
  <em>A deep learning project for classifying breast histopathology images as Benign or Malignant.</em>
</p>

> **Disclaimer:** This project is intended for educational and research purposes. The model is not clinically validated and must not be used for medical diagnosis or treatment decisions.

---

## 📌 Overview

This project develops and evaluates Convolutional Neural Network (CNN) models for binary classification of breast histopathology images into:

- **Benign**
- **Malignant**

The project uses the **BreaKHis Breast Cancer Histopathological Database** and implements an end-to-end deep learning workflow, including:

- Dataset extraction and inspection
- Class distribution analysis
- Histological subtype mapping
- Specimen-level dataset splitting
- Image preprocessing and normalization
- Data augmentation
- CNN model development
- Model training and checkpointing
- Model evaluation and comparison
- Individual image classification using a saved model

Two CNN architectures are implemented and compared:

1. **Baseline CNN** — a conventional convolutional neural network.
2. **Regularized CNN** — a parameter-efficient architecture using Batch Normalization, Global Average Pooling, and Dropout.

The models are evaluated using accuracy, precision, recall, F1-score, AUC, confusion matrices, ROC curves, and precision-recall curves.

---

## 🎯 Project Objectives

- Build a CNN-based breast histopathology image classification pipeline.
- Classify images into Benign and Malignant categories.
- Apply image resizing, normalization, and data augmentation.
- Use specimen-level splitting to reduce potential data leakage.
- Compare baseline and regularized CNN architectures.
- Evaluate model performance using multiple classification metrics.
- Save and reload trained models.
- Perform classification on randomly selected test images.
- Develop a reproducible foundation for medical image analysis research.

---

## 🔬 Research Question

> How does a regularized CNN architecture compare with a baseline CNN for binary classification of breast histopathology images using specimen-level dataset splitting?

The comparison considers multiple evaluation metrics rather than relying only on accuracy.

---

## 📂 Dataset

This project uses the **BreaKHis Breast Cancer Histopathological Database**.

The selected images are organized by histological subtype and microscope magnification.

### Dataset Summary

| Property | Value |
|---|---:|
| Dataset | BreaKHis |
| Task | Binary image classification |
| Image records used | 7,909 |
| Training images | 5,553 |
| Validation images | 961 |
| Test images | 1,395 |
| Unique specimens | 82 |
| Input image size | `224 × 224 × 3` |
| Number of classes | 2 |
| Magnifications | 40X, 100X, 200X, 400X |

### Binary Class Mapping

| Binary Class | Histological Subtypes |
|---|---|
| **Benign** | Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma |
| **Malignant** | Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma |

> **Note:** The subtype-to-binary-class mapping is a project-specific design choice.

### Class Distribution

| Class | Number of Images | Percentage |
|---|---:|---:|
| Benign | 2,480 | 31.36% |
| Malignant | 5,429 | 68.64% |
| **Total** | **7,909** | **100%** |

The dataset is imbalanced toward the Malignant class. Therefore, class-wise precision, recall, and F1-score are reported in addition to overall accuracy.

---

## 🔐 Specimen-Level Data Splitting

Histopathology datasets can contain multiple images originating from the same specimen. Randomly splitting individual images can result in related images appearing in both training and testing subsets.

To reduce this risk, this project uses `GroupShuffleSplit` with specimen identifiers.

### Dataset Split

| Split | Images | Specimens |
|---|---:|---:|
| Training | 5,553 | 57 |
| Validation | 961 | 12 |
| Test | 1,395 | 13 |
| **Total** | **7,909** | **82** |

Specimen-level splitting is intended to reduce overlap between related images across dataset partitions.

> **Note:** The specimen identifier extraction should be independently validated before making strong claims about leakage prevention.

---

## 🏗️ Project Workflow

```text
Dataset ZIP
    │
    ▼
Dataset Extraction and Verification
    │
    ▼
Image Metadata Collection
    │
    ▼
Subtype-to-Binary Class Mapping
    │
    ▼
Specimen-Level Dataset Splitting
    │
    ├── Training Dataset
    ├── Validation Dataset
    └── Test Dataset
    │
    ▼
Image Preprocessing
    │
    ├── RGB Conversion
    ├── Resize to 224 × 224
    └── Pixel Normalization
    │
    ▼
Training Data Augmentation
    │
    ▼
CNN Model Development
    │
    ├── Baseline CNN
    └── Regularized CNN
    │
    ▼
Model Training and Checkpointing
    │
    ▼
Model Evaluation
    │
    ├── Accuracy
    ├── Precision
    ├── Recall
    ├── F1-score
    ├── AUC
    ├── Confusion Matrix
    ├── ROC Curve
    └── Precision-Recall Curve
    │
    ▼
Model Saving and Individual Image Inference
```

---

## ⚙️ Preprocessing and Training Configuration

| Configuration | Value |
|---|---|
| Image size | `224 × 224 × 3` |
| Batch size | `32` |
| Random seed | `42` |
| Optimizer | Adam |
| Initial learning rate | `0.0001` |
| Loss function | Binary cross-entropy |
| Output activation | Sigmoid |
| Image normalization | Pixel values divided by `255` |
| Early stopping | Enabled |
| Learning-rate reduction | Enabled |
| Best-model checkpointing | Enabled |

### Image Preprocessing

The preprocessing pipeline includes:

- Reading image files.
- Decoding images into RGB format.
- Resizing images to `224 × 224` pixels.
- Converting pixel values to floating-point values.
- Normalizing pixel values to the range `[0, 1]`.

### Data Augmentation

The training pipeline uses:

- Random horizontal flipping
- Random rotation
- Random zoom

Data augmentation is applied to the training dataset only.

---

## 🧠 Model Architectures

### 1. Baseline CNN

The baseline CNN contains:

- Three convolutional layers
- Max pooling layers
- A Flatten layer
- A Dense layer with 128 units
- Dropout with a rate of 0.5
- A sigmoid output layer

The baseline CNN contains approximately **12.94 million parameters**.

```text
Input: 224 × 224 × 3
    │
    ▼
Conv2D: 32 filters
    │
    ▼
MaxPooling2D
    │
    ▼
Conv2D: 64 filters
    │
    ▼
MaxPooling2D
    │
    ▼
Conv2D: 128 filters
    │
    ▼
MaxPooling2D
    │
    ▼
Flatten
    │
    ▼
Dense: 128 units
    │
    ▼
Dropout: 0.5
    │
    ▼
Sigmoid Output
```

---

### 2. Regularized CNN

The regularized CNN contains:

- Convolutional layers
- Batch Normalization
- ReLU activation
- Max pooling
- Global Average Pooling
- A Dense layer with 128 units
- Dropout with a rate of 0.5
- A sigmoid output layer

The regularized CNN contains approximately **110,561 total parameters**.

```text
Input: 224 × 224 × 3
    │
    ▼
Conv2D: 32 filters
    │
    ▼
Batch Normalization
    │
    ▼
ReLU Activation
    │
    ▼
MaxPooling2D
    │
    ▼
Conv2D: 64 filters
    │
    ▼
Batch Normalization
    │
    ▼
ReLU Activation
    │
    ▼
MaxPooling2D
    │
    ▼
Conv2D: 128 filters
    │
    ▼
Batch Normalization
    │
    ▼
ReLU Activation
    │
    ▼
MaxPooling2D
    │
    ▼
Global Average Pooling
    │
    ▼
Dense: 128 units
    │
    ▼
Dropout: 0.5
    │
    ▼
Sigmoid Output
```

Global Average Pooling reduces the number of parameters compared with the Flatten-based baseline architecture.

---

## 📊 Experimental Results

The following results were obtained on the reported test dataset containing 1,395 images.

### Baseline CNN vs. Regularized CNN

| Metric | Baseline CNN | Regularized CNN | Difference |
|---|---:|---:|---:|
| Accuracy | 0.7885 | **0.8179** | +0.0294 |
| AUC | 0.7935 | **0.8584** | +0.0649 |
| Loss | 0.4971 | **0.4101** | −0.0870 |
| Precision | 0.8564 | **0.8653** | +0.0089 |
| Recall | 0.8361 | **0.8742** | +0.0381 |

### Results Summary

On the reported test split, the regularized CNN achieved:

- Higher accuracy than the baseline CNN.
- Higher AUC than the baseline CNN.
- Higher precision than the baseline CNN.
- Higher recall than the baseline CNN.
- Lower test loss than the baseline CNN.

These results apply only to the selected test split. They do not establish performance on external datasets or clinical populations.

---

## 📋 Regularized CNN Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Benign | 0.7060 | 0.6894 | 0.6976 | 425 |
| Malignant | 0.8653 | 0.8742 | 0.8697 | 970 |
| **Accuracy** | | | **0.8179** | **1,395** |
| Macro Average | 0.7857 | 0.7818 | 0.7837 | 1,395 |
| Weighted Average | 0.8168 | 0.8179 | 0.8173 | 1,395 |

The class-wise results show different performance levels between Benign and Malignant images. This supports reporting class-specific metrics in addition to aggregate accuracy.

---

## 📈 Evaluation Methods

The project includes the following evaluation methods:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Individual Image Inference

The individual image inference section displays:

- Original image
- True label
- Predicted label
- Benign probability
- Malignant probability

---

## 💾 Saved Artifacts

The project saves trained model checkpoints and evaluation results.

### Example Directory Structure

```text
CNN_Breast_Cancer/
│
├── models/
│   ├── baseline_cnn_best.keras
│   └── regularized_cnn_best.keras
│
├── results/
│   ├── baseline_test_results.csv
│   └── regularized_cnn_test_predictions.csv
│
├── extracted_dataset/
│
├── Dataset.zip
│
└── Breast_Cancer_Model.ipynb
```

The exact files depend on which notebook cells are executed.

---

## 🚀 Installation and Usage

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Google Colab or a compatible local environment

### 1. Clone the Repository

```bash
git clone https://github.com/Ankit-Acharya/Breast_Cancer_CNN.git
cd Breast_Cancer_CNN
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
```

### 3. Run the Notebook

1. Open the notebook in Google Colab or Jupyter.
2. Configure the dataset path.
3. Place the dataset ZIP file in the configured location.
4. Execute the notebook cells in order.
5. Review the training and evaluation results.
6. Save the trained models and evaluation artifacts.
7. Run the individual image inference section.

> Large datasets, model checkpoints, and generated artifacts may be excluded from Git version control because of file size and licensing restrictions.

---

## 🧪 Reproducibility

The project uses a fixed random seed:

```python
SEED = 42
```

The seed is used for NumPy, TensorFlow, and dataset splitting operations.

Exact reproducibility may vary because of:

- Hardware differences
- GPU execution
- TensorFlow versions
- CUDA and cuDNN versions
- Nondeterministic operations
- Dataset ordering
- Runtime configuration

---

## ⚠️ Limitations

### Dataset Limitations

- Evaluation is based on the selected dataset and test split.
- The dataset is imbalanced toward the Malignant class.
- The subtype-to-binary-class mapping is project-specific.
- External dataset validation has not been reported.
- The specimen identifier extraction should be independently verified.

### Model Limitations

- The models are trained from scratch.
- The models have not been compared against a comprehensive set of transfer-learning architectures.
- A single test split does not fully establish model generalization.
- Performance may vary across magnification levels and image sources.
- The classification threshold can affect precision and recall.

### Clinical Limitations

This project is an experimental machine learning implementation. It is not a medical diagnostic system and must not be used to:

- Diagnose breast cancer.
- Rule out breast cancer.
- Guide treatment decisions.
- Replace professional medical evaluation.

Clinical deployment would require extensive independent validation, calibration, prospective evaluation, regulatory assessment, and review by qualified medical professionals.

---

## 🔭 Future Work

- Evaluate transfer learning using ResNet, EfficientNet, and MobileNet.
- Perform systematic hyperparameter optimization.
- Investigate class-weighted training.
- Explore alternative loss functions.
- Evaluate attention-based architectures.
- Compare additional regularization methods.
- Perform group-based cross-validation.
- Analyze performance separately by magnification.
- Evaluate the model on an independent external dataset.
- Investigate classification threshold optimization.
- Implement Grad-CAM visualizations.
- Perform detailed error analysis by subtype and magnification.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow |
| Neural Network API | Keras |
| Data Processing | NumPy, Pandas |
| Image Processing | Pillow |
| Visualization | Matplotlib, Seaborn |
| Evaluation Metrics | Scikit-learn |
| Development Environment | Google Colab |
| Model Format | Keras `.keras` |

---

## 📁 Repository Structure

```text
Breast_Cancer_CNN/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Breast_Cancer_Model.ipynb
│
├── models/
│   └── .gitkeep
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── docs/
    └── methodology.md
```

This is a recommended repository structure for future organization.

---

## 📚 Dataset Attribution

This project uses the BreaKHis breast histopathology image dataset.

Before publishing the repository, include:

- The official dataset citation.
- The dataset source URL.
- Applicable dataset licensing information.
- Citations for external architectures or pretrained models.
- Relevant research papers supporting the methodology.

Do not redistribute the dataset unless redistribution is explicitly permitted by its terms.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

Potential contribution areas include:

- Improved preprocessing methods
- Additional CNN architectures
- More robust evaluation procedures
- Explainable AI visualizations
- Reproducibility improvements
- Documentation enhancements

For substantial changes, open an issue before submitting a pull request.

---

## 📜 License

Choose a license appropriate for your original code and documentation.

Before selecting a license, verify that the dataset and third-party materials permit your intended use and distribution.

---

## 👤 Author

**Ankit Acharya**

- GitHub: [@Ankit-Acharya](https://github.com/Ankit-Acharya)
- Repository: [Breast_Cancer_CNN](https://github.com/Ankit-Acharya/Breast_Cancer_CNN)
- Project: Breast Cancer Histopathology Image Classification
- Focus: Deep Learning, Computer Vision, and Medical Image Analysis

---

## ⭐ Acknowledgment

This project was developed as an educational and research-oriented deep learning experiment using breast histopathology images.

The project focuses on CNN-based image classification, model regularization, evaluation metrics, and reproducible machine learning workflows.
