# Fashion Image Classification on Maryland Polyvore 

## Authors
- Kamand Kalashi
- Sajjad Saed

## Project Overview
This repository showcases a comprehensive exploration of fashion image classification using the Maryland Fashion Dataset. The project is structured into four distinct scenarios, each demonstrating different methodologies and techniques for enhancing image classification capabilities in the fashion domain. The Maryland Fashion Dataset can be accessed [here](https://github.com/AemikaChow/AiDLab-fAshIon-Data/blob/main/Datasets/cleaned-maryland.md).

---

## Scenario 1: CNN from Scratch

### Project Overview
In this scenario, we build a Convolutional Neural Network (CNN) from scratch to classify fashion images based on distinct categories derived from the Maryland Fashion Dataset.

### Key Steps
1. **Environment Setup**: Essential libraries such as TensorFlow and Keras are imported for building and training the CNN.
2. **Data Preparation**: The dataset is mounted, class names retrieved, and sample images displayed.
3. **Image Preprocessing**: Images are resized to 128x128 pixels, converted to grayscale, and pixel values normalized.
4. **Data Augmentation**: Techniques like random rotations and shifts enhance model robustness.
5. **Model Architecture**: A custom CNN is designed with convolutional, pooling, dropout, and dense layers.
6. **Model Compilation**: Categorical crossentropy is used as the loss function with the Adam optimizer.
7. **Model Training**: The model is trained on the training dataset with validation monitoring.
8. **Model Evaluation**: Performance metrics such as accuracy and confusion matrices are computed.

### Results
Two training sessions were conducted, yielding an accuracy of **84%** and **79%**. Notably, the 79% model exhibited greater stability during training.

---

## Scenario 2: Transfer Learning on VGG16

### Project Overview
This scenario employs transfer learning using the VGG16 architecture to enhance image classification in the fashion domain.

### Key Steps
1. **Environment Setup**: Essential libraries are imported, including TensorFlow and Keras.
2. **Data Preparation**: Dataset is mounted, class names retrieved, and sample images displayed.
3. **Image Preprocessing**: Images are resized to 224x224 pixels, and pixel values normalized.
4. **Data Augmentation**: Techniques such as random rotations and flips are applied.
5. **Model Architecture**: The pre-trained VGG16 model is utilized, with additional layers for classification.
6. **Model Compilation**: Categorical crossentropy is used as the loss function with the Adam optimizer.
7. **Model Training**: The model is trained with performance monitoring.
8. **Model Evaluation**: Performance metrics are computed, including accuracy and confusion matrices.

### Results
This project demonstrates the effectiveness of transfer learning in improving classification capabilities.

---

## Scenario 3: Fine-Tuning VGG19

### Project Overview
In this scenario, we focus on fine-tuning the VGG19 architecture for image classification tasks within the fashion domain.

### Key Steps
1. **Environment Setup**: Essential libraries such as TensorFlow and Keras are imported.
2. **Data Preparation**: The dataset is mounted, class names retrieved, and sample images displayed.
3. **Image Preprocessing**: Images are resized to 150x150 pixels, and pixel values normalized.
4. **Data Augmentation**: Techniques to enhance variability and robustness are applied.
5. **Model Architecture**: The pre-trained VGG19 model is utilized, followed by additional layers for classification.
6. **Model Compilation**: Categorical crossentropy is used with the Adam optimizer.
7. **Model Training**: The model is trained on the dataset with performance monitoring.
8. **Model Evaluation**: Performance metrics are computed, including accuracy and confusion matrices.

### Results
This project illustrates the potential of fine-tuning pre-trained models for achieving high accuracy in fashion image classification.

---

## Scenario 4: AutoEncoder

### Project Overview
In this scenario, we implement an autoencoder model to process and reconstruct fashion images from a custom dataset sourced from Polyvore.

### Key Steps
1. **Dataset Preparation**: Fashion images are organized, resized to 128x128 pixels, and normalized.
2. **Autoencoder Architecture**: The architecture includes an encoder and decoder with dropout layers.
3. **Model Compilation**: Adam optimizer and binary cross-entropy loss function are used.
4. **Model Training**: The model is trained with augmented data over 50 epochs.
5. **Evaluation**: The quality of image reconstruction is assessed visually, and a classification model is evaluated using accuracy metrics.
6. **Results Visualization**: Training and validation metrics are plotted, and confusion matrices are generated.

### Conclusions
This project demonstrates the effectiveness of autoencoders in extracting meaningful features from fashion images. Future work could explore integrating the autoencoder with a recommendation engine.

---

## Requirements
To run this project, ensure you have the following libraries installed:
- NumPy
- Pandas
- OpenCV
- TensorFlow
- Keras
- Matplotlib
- Seaborn

---

## Conclusion
Through these four scenarios, this repository showcases a comprehensive approach to fashion image classification, highlighting various methodologies and their effectiveness. Future enhancements will focus on integrating these techniques for improved performance in fashion recommendations.
