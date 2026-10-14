# FACIAL EXPRESSION RECOGNITION USING DEEP LEARNING
## College Mini Project Report

**Submitted by:** [Student Name]  
**Roll Number:** [Roll Number]  
**Programme:** B.Tech — Artificial Intelligence and Machine Learning  
**Department:** [Department]  
**College/University:** [College Name]  
**Academic Session:** [Year]  
**Project Guide:** [Faculty Name]

## Certificate
This is to certify that the mini project titled “Facial Expression Recognition Using Deep Learning” has been prepared by [Student Name] as part of the academic requirements of the B.Tech programme.

Guide Signature: __________  Date: __________

## Declaration
I declare that this report is prepared for academic learning. Dataset sources and terms of use will be acknowledged.

## Acknowledgement
I thank my project guide and department for their guidance. I also acknowledge the developers of Python, TensorFlow/Keras, NumPy, Pillow, and Streamlit.

## Abstract
This project demonstrates image classification using a Convolutional Neural Network (CNN). Given a facial image, the model predicts one of the expression labels represented in its training dataset, such as happy, sad, angry, fear, surprise, disgust, or neutral. Images are resized and normalized before model inference. A Streamlit interface lets a user upload an image and view the predicted label and class scores. This is an educational image-classification project; expression labels do not reliably reveal internal emotional states, and predictions may be wrong.

## 1. Introduction
CNNs can learn visual patterns from labeled images. This project applies a CNN to a facial-expression dataset to demonstrate image preprocessing, training, evaluation, and inference.

## 2. Problem Statement
Build a simple application that classifies an input facial image into one of the labels used by the training dataset and presents the result in an accessible interface.

## 3. Objectives
- Understand CNN basics.
- Learn image resizing and normalization.
- Train an image classifier with TensorFlow/Keras.
- Build an interactive Streamlit app.
- Evaluate performance on validation data.

## 4. Scope and Responsible Use
The system predicts dataset labels, not a person's actual internal emotion, intention, honesty, or personality. Accuracy may vary with lighting, pose, image quality, and dataset composition. Do not use it for high-stakes decisions or surveillance. Use images only with permission.

## 5. Existing and Proposed System
Existing approach: manually browse or label image categories. Proposed approach: use a CNN trained on labeled examples to predict a dataset expression label for an uploaded image.

## 6. Tools and Technologies
Python, TensorFlow/Keras, CNN, NumPy, Pillow, Streamlit, and a labeled image dataset.

## 7. Requirements
Hardware: laptop/desktop with sufficient RAM; a GPU may speed training. Software: compatible Python/TensorFlow version, pip, and a browser.

## 8. Dataset
No dataset is bundled. Obtain a dataset from a source whose terms permit your intended use. Record its source, license, class counts, and preprocessing. Arrange images under `data/fer_train/<class-name>/`; optionally use `data/fer_validation/`.

## 9. Methodology
1. Load labeled images from class folders.
2. Resize to 48×48 and convert to grayscale.
3. Scale pixel values from 0–255 to 0–1.
4. Pass images through convolution, batch-normalization, and pooling layers.
5. Train dense output layers with Adam and sparse categorical cross-entropy.
6. Evaluate on validation data.
7. Save the model and use it in Streamlit.

## 10. Architecture
Dataset → Preprocessing → CNN Feature Learning → Softmax Class Scores → Predicted Dataset Label → Streamlit Interface.

## 11. Algorithm
1. Start.
2. Load labeled images.
3. Resize and normalize images.
4. Initialize CNN.
5. Train using labeled examples.
6. Evaluate on validation examples.
7. Save model.
8. Accept an uploaded image.
9. Apply the same preprocessing.
10. Display the highest-scoring dataset label.
11. Stop.

## 12. Implementation
`train_model.py` builds and trains the CNN. `app.py` loads the trained model, preprocesses an uploaded image, and displays the predicted label and class scores. The model file is created after training and is not included in this ZIP.

## 13. Testing Plan
- Check dataset folder recognition.
- Confirm training creates `models/emotion_model.keras`.
- Upload a supported image and verify a label and score chart appear.
- Test image formats and sizes.
- Record validation loss, accuracy, and failure cases honestly.

## 14. Results
Fill this section after running the project:
- Dataset/source/license: [Fill in]
- Training images: [Fill in]
- Validation images: [Fill in]
- Epochs completed: [Fill in]
- Measured validation accuracy: [Fill in]
- Observations: [Fill in]

## 15. Advantages
- Demonstrates a practical CNN workflow.
- Has a simple interactive interface.
- Can be extended with better data and evaluation.

## 16. Limitations
- Depends on dataset quality and diversity.
- Lighting, pose, and occlusion can affect results.
- Expression labels do not establish a person's internal emotional state.
- May not generalize to different settings.

## 17. Future Scope
Add data augmentation, class balancing, confusion matrix, per-class metrics, better preprocessing, and a clear consent/privacy notice.

## 18. Conclusion
The project demonstrates a CNN-based image-classification workflow and a simple web interface. It provides practice in preprocessing, training, validation, and inference. Results should be interpreted cautiously.

## 19. References
1. TensorFlow: https://www.tensorflow.org/
2. Keras: https://keras.io/
3. Streamlit: https://docs.streamlit.io/
4. NumPy: https://numpy.org/doc/
5. Pillow: https://pillow.readthedocs.io/

Before submission, replace placeholders, use a permitted dataset, run the model, add genuine screenshots, and follow your college format.
