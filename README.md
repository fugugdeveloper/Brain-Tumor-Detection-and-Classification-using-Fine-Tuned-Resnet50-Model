
# Brain Tumor Detection and Classification Using Fine-Tuned ResNet50

This repository contains the implementation and analysis of a deep learning model designed to classify brain tumors from MRI images into four distinct categories: glioma, meningioma, pituitary tumor, and no tumor. The study focuses on leveraging the ResNet50 architecture, comparing its base performance against a fine-tuned version adapted through transfer learning techniques. The goal of this work is to enhance classification accuracy and reliability for potential clinical applications.

## Project Overview

Accurate and timely diagnosis of brain tumors is essential for effective treatment planning and improved patient outcomes. Manual analysis of MRI images by radiologists, although standard, can be prone to human error and delays. This project introduces an automated solution that applies state-of-the-art deep learning methods to classify brain tumors with high precision and recall.

Key contributions include:
- A comparative study between a base ResNet50 model and a fine-tuned version tailored for medical image classification.
- Advanced preprocessing techniques to enhance image contrast and clarity using CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gaussian blur sharpening.
- Comprehensive evaluation of model performance using multiple classification metrics such as accuracy, precision, recall, F1-score, and confusion matrix analysis.
- Integration of Grad-CAM visualizations to support interpretability and clinical trust.
- Deployment of a lightweight web application using Gradio to enable real-time predictions from user-uploaded MRI images.

## Dataset

The model was trained on a combined dataset of over 10,000 MRI scans sourced from publicly available Kaggle repositories. The dataset comprises four balanced classes:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

All images were converted to grayscale and resized to 224x224 pixels. After preprocessing and augmentation, the dataset was split into training, validation, and testing sets to ensure proper evaluation of the model's generalization.

## Data Preprocessing and Augmentation

The preprocessing stage played a crucial role in enhancing the MRI image quality. Key steps included:
- Conversion of grayscale images to RGB to match the input format of ResNet50.
- Application of CLAHE to improve local contrast in medical images.
- Use of Gaussian-blur-based sharpening to enhance structural features.
- Data augmentation techniques such as rotation, zooming, shifting, flipping, and shearing to increase data variability and reduce overfitting.

## Model Architecture and Training

The base model used was ResNet50, pre-trained on the ImageNet dataset. A custom classification head was added, comprising:
- Global average pooling layer to reduce feature dimensions.
- Two fully connected layers (512 and 128 units) with ReLU activation.
- Batch normalization and dropout layers for regularization.
- Final softmax output layer with four neurons corresponding to the four target classes.

Training was conducted in two phases:
1. Transfer Learning: Freezing the ResNet50 base and training the custom top layers.
2. Fine-Tuning: Unfreezing selected layers of the base model and retraining with a lower learning rate to refine feature extraction for domain-specific images.

Model performance was monitored using callbacks such as EarlyStopping and ReduceLROnPlateau. The training and validation results were saved and visualized to analyze convergence and detect signs of overfitting.

## Performance Comparison

The base model achieved:
- Overall accuracy: 87%
- Average F1-score: 0.87
- Recall for glioma and meningioma classes was notably lower due to class confusion.

The fine-tuned model demonstrated significant improvements:
- Overall accuracy: 98%
- Macro and weighted F1-scores: 0.98
- Near-perfect classification performance for all classes, especially glioma and pituitary tumors.

The improvement highlights the benefits of domain-specific fine-tuning and advanced preprocessing.

## Model Interpretability with Grad-CAM

To ensure transparency and increase trust in model predictions, Grad-CAM (Gradient-weighted Class Activation Mapping) was employed. Grad-CAM heatmaps highlighted areas in MRI scans that contributed most to the model’s predictions, providing visual explanations for clinical verification.

## Web Application Deployment

A simple, user-friendly web interface was developed using Gradio. This application allows users to upload MRI images, receive classification results with class-wise confidence scores, and view Grad-CAM visualizations. It demonstrates the practical applicability of the trained model in real-world medical workflows.

## Limitations and Future Directions

Despite strong performance, the model has limitations:
- It uses 2D axial slices, limiting 3D tumor context analysis.
- Dataset was curated and pre-cropped, not reflective of raw clinical images.
- Further evaluation is required on external datasets to validate generalization.
- Additional explainable AI techniques can enhance model transparency.

Future enhancements include:
- Integration of 3D convolutional networks.
- Use of multi-modal data (e.g., PET, CT, patient metadata).
- Deployment on mobile devices using model quantization.
- Clinical trials and validation with expert feedback.

## References

- Masoud Nickparvar - Brain Tumor MRI Dataset (Kaggle)
- Bilalakgz - Brain Tumor MRI Dataset (Kaggle)
- Sajjad et al. (2019), Afshar et al. (2020), Chowdhury et al. (2021), among others.

## Summary

This project demonstrates that a fine-tuned ResNet50 model can significantly outperform a base model in classifying brain tumors from MRI scans. With an accuracy of 98% and strong interpretability via Grad-CAM, the system shows promise for clinical support in brain tumor diagnostics.
