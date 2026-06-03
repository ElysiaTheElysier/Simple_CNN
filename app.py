import os
import streamlit as st
from PIL import Image
from utils.prediction import load_cnn_model, predict_image, get_class_names

# --- Page Config ---
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Gradient header */
    .main-header {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 3rem;
    }
    .main-header p {
        color: #e0e0e0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Rounded cards */
    .stCard {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #333333;
    }
    .stCard h3, .stCard h4 {
        color: #1f2937;
    }
    
    /* Result box */
    .result-box {
        background-color: #e0f2fe;
        border-left: 5px solid #0284c7;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        color: #0c4a6e;
    }
    
    /* Progress bar override for visibility */
    .stProgress > div > div > div > div {
        background-color: #0ea5e9;
    }
    
    /* Dark text on light background overall */
    .css-1d391kg {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>CNN Image Classifier with TensorFlow</h1>
    <p>Upload an image and let a trained CNN model classify it into one of the CIFAR-10 categories.</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ Project Information")
    st.markdown("""
    - **Model:** Simple CNN
    - **Framework:** TensorFlow / Keras
    - **Dataset:** CIFAR-10
    - **Input size:** 32x32 RGB
    - **Classes:** 10
    
    **Workflow:**
    1. Upload image
    2. Preprocess
    3. CNN model
    4. Prediction
    """)
    st.divider()
    st.markdown("*Developed as a student project demo.*")

# --- Model Loading ---
@st.cache_resource
def get_model():
    model_path = os.path.join("models", "cnn_cifar10_model.keras")
    if not os.path.exists(model_path):
        return None
    return load_cnn_model(model_path)

model = get_model()

# Check if model exists
if model is None:
    st.warning("⚠️ Model file not found. Please run: `python train_model.py` in your terminal to train and save the model.")

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            st.markdown("**Image Details:**")
            st.write(f"- Original Size: {image.size[0]}x{image.size[1]}")
            st.write(f"- Mode: {image.mode}")
            st.write(f"- File Type: {uploaded_file.type}")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            uploaded_file = None
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("🎯 Prediction")
    
    st.info("ℹ️ The uploaded image will be resized to 32x32 pixels because the CNN model was trained on CIFAR-10 images of that exact size.")
    
    if uploaded_file is not None:
        if st.button("Predict Image", type="primary", use_container_width=True):
            if model is None:
                st.error("Cannot predict: Model not found.")
            else:
                with st.spinner("Analyzing image..."):
                    try:
                        results = predict_image(model, image, top_k=3)
                        
                        top_class = results[0]['class_name']
                        top_conf = results[0]['confidence'] * 100
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h2 style="margin:0; color:#0369a1;">Top Prediction: {top_class.capitalize()}</h2>
                            <p style="margin:0; font-size:1.2rem;">Confidence: {top_conf:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### Top 3 Predictions:")
                        for res in results:
                            class_name = res['class_name'].capitalize()
                            conf = res['confidence']
                            
                            st.write(f"{class_name} ({conf*100:.2f}%)")
                            st.progress(float(conf))
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
    else:
        st.write("Upload an image and click predict to see the results.")
        
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- How CNN Works Section ---
st.subheader("🧠 How the CNN works")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stCard" style="height:150px; text-align:center;">', unsafe_allow_html=True)
    st.markdown("#### Convolution")
    st.markdown("Extracts visual features like edges and textures.")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stCard" style="height:150px; text-align:center;">', unsafe_allow_html=True)
    st.markdown("#### ReLU")
    st.markdown("Adds non-linearity to learn complex patterns.")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stCard" style="height:150px; text-align:center;">', unsafe_allow_html=True)
    st.markdown("#### Pooling")
    st.markdown("Reduces feature map size, retaining important info.")
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stCard" style="height:150px; text-align:center;">', unsafe_allow_html=True)
    st.markdown("#### Softmax")
    st.markdown("Outputs probabilities for each class (0 to 1).")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- Training Results Section ---
st.subheader("📊 Training Results")
acc_path = os.path.join("assets", "training_accuracy.png")
loss_path = os.path.join("assets", "training_loss.png")

if os.path.exists(acc_path) and os.path.exists(loss_path):
    rc1, rc2 = st.columns(2)
    with rc1:
        st.image(acc_path, caption="Training & Validation Accuracy")
    with rc2:
        st.image(loss_path, caption="Training & Validation Loss")
else:
    st.info("Training charts will appear after running `python train_model.py`")

st.divider()

# --- Disclaimer ---
st.caption("⚠️ **Disclaimer:** This demo model is trained on CIFAR-10, so it works best for images similar to CIFAR-10 classes. It may predict incorrectly for blurry images, unusual angles, or objects outside the 10 classes.")
