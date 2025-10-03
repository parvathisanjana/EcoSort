#!/usr/bin/env python3
"""
Streamlit Waste Classification App
A beautiful web interface for waste classification using deep learning
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pickle
import os
from corrected_classification import get_recycling_guidance, DETAILED_CLASSIFICATION

# Page configuration
st.set_page_config(
    page_title="♻️ Waste Classification AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #2E8B57, #32CD32);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .non-recyclable-card {
        border-left-color: #dc3545;
    }
    .recyclable-card {
        border-left-color: #28a745;
    }
    .confidence-bar {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #20c997);
        transition: width 0.3s ease;
    }
    .category-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .category-item {
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-size: 0.9rem;
    }
    .recyclable-category {
        background: #d4edda;
        color: #155724;
    }
    .non-recyclable-category {
        background: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and label encoder"""
    model_path = "waste_classifier.h5"
    encoder_path = "label_encoder.pkl"
    categories_path = "category_names.pkl"
    
    if not all(os.path.exists(path) for path in [model_path, encoder_path, categories_path]):
        st.error("❌ Model files not found! Please ensure the following files exist:")
        st.code("""
waste_classifier.h5
label_encoder.pkl
category_names.pkl
        """)
        st.stop()
    
    try:
        # Load model
        model = keras.models.load_model(model_path)
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load category names
        with open(categories_path, 'rb') as f:
            category_names = pickle.load(f)
        
        return model, label_encoder, category_names
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model inference"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_waste_item(model, label_encoder, category_names, image):
    """Predict specific waste item"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Get top 5 predictions
        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
        top_5_predictions = []
        
        for idx in top_5_indices:
            category_name = category_names[idx]
            confidence = prediction[0][idx]
            top_5_predictions.append({
                'category': category_name,
                'confidence': float(confidence),
                'confidence_percent': float(confidence * 100)
            })
        
        # Get the top prediction
        top_prediction = top_5_predictions[0]
        
        # Get recycling information
        recycling_info = DETAILED_CLASSIFICATION.get(top_prediction['category'], {
            'category': 'unknown',
            'type': 'Unknown',
            'reason': 'Category not found in database',
            'disposal': 'Check local recycling guidelines'
        })
        
        return {
            'top_prediction': top_prediction,
            'top_5_predictions': top_5_predictions,
            'recycling_info': recycling_info,
            'detailed_guidance': get_recycling_guidance(top_prediction['category'])
        }
        
    except Exception as e:
        return {"error": str(e)}

def format_category_name(category_name):
    """Format category name for display"""
    return category_name.replace('_', ' ').title()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>♻️ Waste Classification AI</h1>
        <p>Upload an image to identify waste items and get recycling guidance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, label_encoder, category_names = load_model()
    
    # Sidebar
    st.sidebar.title("📊 Model Information")
    st.sidebar.write(f"**Total Categories:** {len(category_names)}")
    st.sidebar.write(f"**Model Type:** MobileNetV2 Transfer Learning")
    st.sidebar.write(f"**Input Size:** 224x224 pixels")
    
    # Category breakdown
    st.sidebar.title("🗂️ Waste Categories")
    
    recyclable_categories = []
    non_recyclable_categories = []
    
    for category in category_names:
        recycling_info = DETAILED_CLASSIFICATION.get(category, {})
        if recycling_info.get('category') == 'recyclable':
            recyclable_categories.append(category)
        else:
            non_recyclable_categories.append(category)
    
    st.sidebar.write(f"**Recyclable:** {len(recyclable_categories)} categories")
    st.sidebar.write(f"**Non-Recyclable:** {len(non_recyclable_categories)} categories")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.title("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of waste to classify"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔍 Classify Waste", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    result = predict_waste_item(model, label_encoder, category_names, image)
                
                if "error" not in result:
                    # Display results
                    with col2:
                        st.title("🎯 Classification Results")
                        
                        # Top prediction
                        top = result['top_prediction']
                        recycling = result['recycling_info']
                        
                        # Determine card style
                        card_class = "recyclable-card" if recycling['category'] == 'recyclable' else "non-recyclable-card"
                        icon = "♻️" if recycling['category'] == 'recyclable' else "🚫"
                        
                        st.markdown(f"""
                        <div class="prediction-card {card_class}">
                            <h2>{icon} {format_category_name(top['category'])}</h2>
                            <p><strong>Type:</strong> {recycling['type']}</p>
                            <p><strong>Classification:</strong> {recycling['category'].upper()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.write("**Confidence:**")
                        confidence_percent = top['confidence_percent']
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence_percent}%"></div>
                        </div>
                        <p>{confidence_percent:.1f}%</p>
                        """, unsafe_allow_html=True)
                        
                        # Recycling guidance
                        st.subheader("📖 Recycling Guidance")
                        st.markdown(result['detailed_guidance'])
                        
                        # Top 5 predictions
                        st.subheader("🏆 Top 5 Predictions")
                        for i, pred in enumerate(result['top_5_predictions'], 1):
                            pred_recycling = DETAILED_CLASSIFICATION.get(pred['category'], {})
                            pred_icon = "♻️" if pred_recycling.get('category') == 'recyclable' else "🚫"
                            st.write(f"{i}. {pred_icon} {format_category_name(pred['category'])} ({pred['confidence_percent']:.1f}%)")
                
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 Powered by Deep Learning & Computer Vision</p>
        <p>Built with Streamlit, TensorFlow, and MobileNetV2</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

