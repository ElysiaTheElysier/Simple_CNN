# CNN Image Classifier

A simple but polished demo project for image classification using a TensorFlow/Keras Convolutional Neural Network (CNN) and Streamlit. The project is structured using Jupyter Notebooks for interactive execution and learning.

## Project Structure
```text
tensorflow_cnn_streamlit_demo/
├── app.ipynb                # Notebook to generate and run Streamlit UI
├── train_model.ipynb        # Notebook to train the CNN model
├── utils/
│   └── prediction.ipynb     # Notebook to generate utility functions
├── requirements.txt         # Project dependencies
└── README.md
```
*(Note: `app.py`, `utils/prediction.py`, `models/`, and `assets/` are automatically generated when you run the notebooks).*

## Setup Instructions

1. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

2. **Generate Prediction Utilities:**
Open `utils/prediction.ipynb` and run all cells. This will automatically generate the `utils/prediction.py` file which is needed for the Streamlit app.

3. **Train the Model:**
Open `train_model.ipynb` and run all cells. This will:
- Download the CIFAR-10 dataset.
- Build an enhanced CNN model (with Data Augmentation, 3 Conv2D layers, and Dropout).
- Train the model for up to 30 epochs (using Early Stopping to prevent overfitting).
- Save the trained model to `models/cnn_cifar10_model.keras`.
- Generate training metrics charts in the `assets/` folder.

4. **Run the Streamlit App:**
Open `app.ipynb` and run all cells. This will generate `app.py` and execute the Streamlit server automatically. Alternatively, if `app.py` has already been generated, you can directly run:
```bash
streamlit run app.py
```

## Expected Local URL
The application will be accessible at:
http://localhost:8501

## Model Details
- **Dataset:** CIFAR-10 (60,000 32x32 color images in 10 classes)
- **Architecture:** Convolutional Neural Network (CNN) with Data Augmentation (RandomFlip, RandomRotation), 3x Conv2D layers (32, 64, 128 filters), MaxPooling, Dense (128 units), and Dropout (0.4).
- **Training Strategy:** Max 30 epochs with Early Stopping (patience=4) based on validation loss.
- **Preprocessing:** Images uploaded via the app are resized to 32x32 pixels, converted to RGB, and normalized to 0-1 scale.

**Note:** This app is designed for demonstration purposes. It performs best on images resembling the CIFAR-10 dataset classes.
