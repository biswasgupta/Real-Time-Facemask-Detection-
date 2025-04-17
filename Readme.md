# Real-Time Face Mask Detection

This project implements a real-time face mask detection system that classifies faces into three categories: **with_mask**, **without_mask**, and **mask_weared_incorrect**. The system leverages an ensemble approach by integrating multiple deep learning models to improve prediction accuracy and robustness.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology and Report Highlights](#methodology-and-report-highlights)
- [Contributing](#contributing)
- [License](#license)

## Overview
This system is designed to detect face masks in real time using a combination of pre-trained models including ResNet18, ResNet50, and YOLOv8. The ensemble approach uses majority voting to produce reliable predictions by compensating for the limitations of individual models. The project also integrates classical face detection techniques (using a Caffe-based detector) when required.

## Features
- **Real-Time Detection:** Process webcam feeds to detect faces and classify mask status on the fly.
- **Multiple Model Support:** Choose between ResNet50, ResNet18, YOLOv8, or an ensemble approach for predictions.
- **Image Processing:** Process and annotate static images for mask detection.
- **Ensemble Voting:** Combine predictions from multiple models to improve accuracy.
- **User-Friendly CLI:** Simple command-line interface for selecting detection modes and processing images.

## Project Structure
Final Submission Folder/
├── realtime.py               # Main file for real-time detection and CLI interface
├── models.ipynb              # Notebook containing training and evaluation scripts
├── yolo_v8.pt                # Pre-trained YOLOv8 model used for face detection and mask classification
└── saved_models/             # Directory containing saved model weights
    ├── full_model_resnet18.pth
    ├── full_model_resnet50.pth
    ├── model_resnet50_state_dict.pth
    └── model_resnet18_state_dict.pth


- **realtime.py:**  
  This file initializes the `FaceMaskDetector` class which loads the selected model(s) based on the specified `model_choice`. It provides functionality to:
  - Download necessary DNN files for face detection.
  - Perform face detection using either the Caffe-based detector or YOLOv8.
  - Process individual image files and live webcam feeds.
  - Run a simple command-line interface (CLI) that allows users to choose between real-time detection, image processing, and model selection.

- **models.ipynb:**  
  Contains the scripts used for training and evaluating the models used in this project.

- **yolo_v8.pt:**  
  The YOLOv8 model file is used in `realtime.py` to perform fast and accurate face detection as well as mask classification.

- **saved_models Directory:**  
  Contains the pre-trained weights and state dictionaries for the ResNet18 and ResNet50 models used in the project.

## Installation
Make sure you have Python 3.8 or later installed. Install the required libraries using pip
requirements.txt


## Usage
To run the real-time face mask detection system, execute the following command:

python realtime.py

Once running, you will be presented with a simple CLI with the following options:

1. **Real-time detection (webcam):** Start processing the live video feed from your webcam.  
2. **Process an image file:** Provide the path to an image file for mask detection.  
3. **Change model choice:** Switch between the available models (`resnet50`, `resnet18`, `yolo`, or `ensemble`).  
4. **Exit:** Terminate the application.

Follow the on-screen instructions to test different functionalities.

## Methodology and Report Highlights
This project is based on the detailed study documented in the final project report. Key highlights include:

- **Ensemble Strategy:**  
  Combining predictions from ResNet18, ResNet50, and YOLOv8 using majority voting to improve reliability. In the event of a tie, the YOLOv8 prediction is prioritized due to its robust real-time detection capabilities.

- **Data Preparation:**  
  The models were fine-tuned on a face mask detection dataset (sourced from Kaggle) with images resized and normalized to 226×226 pixels. Data augmentation was also applied to improve model robustness.

- **Model Training:**  
  - **ResNet18 and ResNet50:** Utilized transfer learning with modifications to the final fully connected layers. Techniques such as batch normalization, dropout, and focal loss were employed to tackle class imbalance and reduce overfitting.
  - **YOLOv8:** Trained for simultaneous face detection and mask classification, optimized for real-time performance.

- **Real-Time Inference Pipeline:**  
  Captures video frames, detects faces using either a Caffe-based detector or YOLOv8, and annotates the frames with mask detection results. This is implemented in the `realtime.py` file.

These details are discussed further in the project report included in the repository.

