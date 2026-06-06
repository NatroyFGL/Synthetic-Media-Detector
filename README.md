# Synthetic Media Detection — CIFAKE Binary Classifier

A binary image classifier that distinguishes real photographs from AI-generated images, trained on the CIFAKE dataset (60,000 images). Built as part of an exploration into synthetic media detection.

## Dataset
CIFAKE from Kaggle — 32x32 RGB images split into REAL (CIFAR-10 photos) and FAKE (Stable Diffusion generated) classes, with separate train and test folders.

## How it works
Images are flattened into feature vectors, normalized, and scaled before being fed into a Logistic Regression classifier. The model is evaluated on a held-out test set using precision, recall, F1-score, and a confusion matrix.

## Usage
Install dependencies: pip install -r requirements.txt

Download CIFAKE from Kaggle and place the train/ and test/ folders in the project root, then run: python classify.py

To predict a single image, uncomment and call predict_image("your_image.jpg") at the bottom of classify.py.

## Results
Achieves approximately 75-76% accuracy on the test set.
