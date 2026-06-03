import numpy as np
import tensorflow as tf
from PIL import Image

def get_class_names():
    """Return the CIFAR-10 class names."""
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

def load_cnn_model(model_path):
    """Load the trained CNN model from the given path."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """
    Preprocess the image for the model.
    - Convert to RGB
    - Resize to 32x32
    - Convert to numpy array
    - Normalize by dividing by 255.0
    - Expand dimension to shape (1, 32, 32, 3)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize((32, 32))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, image, top_k=3):
    """
    Predict the class of the image using the given model.
    Returns the top_k results.
    """
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]
    
    class_names = get_class_names()
    
    # Get indices of top_k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for i in top_indices:
        results.append({
            "class_name": class_names[i],
            "confidence": float(predictions[i])
        })
        
    return results
