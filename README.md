# CNN Image Classifier

A simple but polished demo project for image classification using a TensorFlow/Keras Convolutional Neural Network (CNN) and Streamlit.

## Project Structure
```
tensorflow_cnn_streamlit_demo/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── models/
│   └── cnn_cifar10_model.keras
├── assets/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── sample_predictions.png
└── utils/
    ├── __init__.py
    └── prediction.py
```

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the Model:
The model needs to be trained and saved before the app can make predictions.
```bash
python train_model.py --epochs 5
```
This will download the CIFAR-10 dataset (if not already cached), train the CNN model, and save the resulting model to `models/cnn_cifar10_model.keras`. It also generates training charts in the `assets/` folder.

3. Run the Streamlit App:
```bash
streamlit run app.py
```

## Expected Local URL
The application will be accessible at:
http://localhost:8501

## Explanation
- **Dataset:** CIFAR-10 (contains 60,000 32x32 color images in 10 classes)
- **Model:** Convolutional Neural Network (CNN) built with TensorFlow/Keras
- **Preprocessing:** Images uploaded via the app are resized to 32x32 pixels, converted to RGB, and normalized (scaled to 0-1) to match the training data.
- **Output:** The model outputs probabilities for each of the 10 classes using a softmax activation function.

**Note:** This app is designed for a classroom demo, not production-level accuracy. It performs best on images resembling the CIFAR-10 dataset classes.
