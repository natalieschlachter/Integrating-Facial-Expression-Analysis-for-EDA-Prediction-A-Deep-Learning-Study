# Integrating-Facial-Expression-Analysis-for-EDA-Prediction-A-Deep-Learning-Study


## Overview

This project investigates whether **ElectroDermal Activity (EDA)** can
be predicted from facial video sequences using deep learning models.

We evaluate multiple architectures --- including **PhysNet** and
**DeepPhys** --- on the **UBFC-Phys dataset**, a multimodal dataset
containing synchronized:

-   Facial video recordings\
-   Blood Volume Pulse (BVP) signals\
-   ElectroDermal Activity (EDA) signals

The goal is to assess whether state-of-the-art remote physiological
models can generalize to EDA prediction.

------------------------------------------------------------------------

## Research Questions

-   Can EDA be predicted directly from facial expressions?
-   Does integrating BVP signals improve performance?
-   How well do PhysNet and DeepPhys generalize to this task?
-   What are the limitations of current multimodal approaches?

------------------------------------------------------------------------

## Dataset

**UBFC-Phys**

-   56 participants\
-   3 tasks per participant:
    -   Resting
    -   Speech task
    -   Arithmetic stress task
-   Video: 35 FPS → resampled to 4 FPS\
-   EDA: 4 Hz\
-   BVP: 64 Hz → downsampled to 4 Hz

Train / Validation / Test split: - 35 / 10 / 11 subjects

Each participant contributes 2160 synchronized frames across tasks.

------------------------------------------------------------------------

## Implemented Models

### 1. Naive Baseline

Predicts the mean EDA value from the training set.

### 2. Sequential CNN

-   CNN for spatial feature extraction\
-   LSTM for temporal modeling\
-   Fully connected layer for EDA prediction

### 3. Sequential CNN + BVP

-   CNN for video features\
-   BVP embedding (32D)\
-   Feature concatenation → LSTM → prediction

### 4. PhysNet (Modified)

-   Spatio-temporal CNN architecture\
-   Originally designed for rPPG extraction\
-   Extended to integrate BVP and predict EDA

### 5. DeepPhys

-   Convolutional Attention Network\
-   Motion model + appearance model\
-   Attention masks for physiologically relevant regions

------------------------------------------------------------------------

## Evaluation Metrics

-   **Mean Squared Error (MSE)** --- prediction accuracy\
-   **Negative Pearson Correlation (NPC)** --- alignment with true
    signal trends

Important finding:\
Low MSE does not imply meaningful learning --- most models tended to
predict values near the mean, resulting in low correlation.

------------------------------------------------------------------------

## Experiments

### Data Normalization

-   Scaled signals to \[0,1\]
-   Stabilized gradients
-   Did not significantly improve generalization

### Data Augmentation

Applied with probability 0.7: - Random rotation - Color jitter -
Horizontal flip - Perspective distortion

Result: - Improved DeepPhys generalization slightly - Limited impact on
other models

### Face Cropping

-   Used `face_recognition` for facial region isolation
-   Reduced background noise
-   Did not significantly improve correlation

------------------------------------------------------------------------

## Key Findings

-   All models struggled to capture meaningful EDA fluctuations.
-   Predictions were biased toward the dataset mean.
-   DeepPhys showed the best generalization under augmentation.
-   BVP integration did not significantly improve performance.
-   Major limitation: small and homogeneous dataset.

------------------------------------------------------------------------

## Limitations

-   Only 56 participants (gender imbalance)
-   Reduced temporal resolution (4 FPS)
-   No advanced BVP filtering or feature engineering
-   Overfitting across architectures
-   Noisy EDA signals

------------------------------------------------------------------------

## Future Work

-   Integrate OpenFace for facial action unit extraction\
-   Apply advanced BVP preprocessing and filtering\
-   Increase dataset diversity\
-   Use higher temporal resolution\
-   Explore alternative loss functions\
-   Improve multimodal fusion strategies

------------------------------------------------------------------------

## Ethical Considerations

EDA is used in monitoring: - Anxiety - ADHD - Autism - Alzheimer's -
Schizophrenia

Healthcare applications require: - High interpretability - Careful
handling of false negatives - GDPR / HIPAA compliance

------------------------------------------------------------------------

## Tech Stack

-   Python\
-   PyTorch\
-   OpenCV\
-   face_recognition\
-   NumPy\
-   Matplotlib

------------------------------------------------------------------------

## Conclusion

Despite implementing state-of-the-art architectures, predicting EDA from
facial video remains highly challenging.

This study highlights the difficulty of translating controlled
physiological signal models into robust, real-world healthcare
applications.
