# Viva Questions and Answers

**1. What is the aim?** Demonstrate CNN-based classification of facial-expression image labels.

**2. What is deep learning?** A branch of machine learning that uses multi-layer neural networks to learn patterns from data.

**3. What is a CNN?** A neural network architecture often used to learn spatial patterns in images.

**4. Why resize images to 48×48?** The model uses a fixed input shape so images have consistent dimensions.

**5. Why grayscale?** It reduces image channels and matches the model's input shape.

**6. What is normalization?** Scaling pixel values from 0–255 to 0–1.

**7. What is softmax?** A function that converts output logits into class scores that sum to one.

**8. What is overfitting?** When a model fits training examples too closely and performs poorly on unseen examples.

**9. What is validation accuracy?** The proportion of validation examples classified correctly; measure it during evaluation rather than guessing.

**10. Which libraries are used?** TensorFlow/Keras, NumPy, Pillow, and Streamlit.

**11. Do facial expressions prove real emotions?** No. Expression labels do not reliably reveal a person's internal emotional state.

**12. What are limitations?** Dataset bias, lighting and pose variation, class imbalance, limited generalization, and imperfect labels.

**13. How can it be improved?** Use a permitted diverse dataset, data augmentation, confusion matrix, per-class metrics, and held-out testing.

**14. Why is a dataset required?** The CNN learns visual patterns from labeled examples during training.
