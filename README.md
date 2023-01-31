<h1 align="center">Machine Learning And Algorithms Project </h1>

## Introduction
This project aims to build a Facial Emotion Recognition system using the OpenCV library and Convolutional Neural Network (CNN) algorithms. The system will be trained on a dataset of images of faces with annotations of emotions to predict the emotions of new unseen images.

## Technical Details
- OpenCV: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It will be used for image pre-processing and feature extraction.
- CNN: Convolutional Neural Networks are a type of deep learning models specifically designed for image classification tasks. They will be used for emotion recognition in this project.

## Getting Started

### Requirements
- OpenCV 4.4.0 or higher
- Python 3.7 or higher
- TensorFlow 2.3 or higher
- Keras 2.4 or higher

### Setup
1. Clone the repository<br>
```
git clone https://github.com/AmanSinghBhogal/facial-emotion-recognition.git
```
2. Install the required packages
```
pip install -r requirements.txt
```
### Running the project
1. Train the model
```
python train.py
```
2. Test the model
```
python test.py [path to test image]
```
### Data
The dataset used for training the model can be found in the `data` folder. It contains images of faces with annotations of emotions.

## Conclusion
This project serves as a starting point for building a Facial Emotion Recognition system using OpenCV and CNN algorithms. It can be further improved by adding more data and fine-tuning the model.
