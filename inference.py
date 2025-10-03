#!/usr/bin/env python3
"""
Enhanced Waste Classification Inference Script
Identifies specific waste items and provides detailed recycling information
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pickle
from corrected_classification import get_recycling_guidance, DETAILED_CLASSIFICATION

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

def load_model():
    """Load the model and label encoder"""
    model_path = "waste_classifier.h5"
    encoder_path = "label_encoder.pkl"
    categories_path = "category_names.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return None, None, None
    
    if not os.path.exists(encoder_path):
        print(f"Error: Label encoder not found: {encoder_path}")
        return None, None, None
    
    if not os.path.exists(categories_path):
        print(f"Error: Category names not found: {categories_path}")
        return None, None, None
    
    try:
        # Load model
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load category names
        with open(categories_path, 'rb') as f:
            category_names = pickle.load(f)
        
        return model, label_encoder, category_names
    
    except Exception as e:
        print(f"Error loading enhanced model: {e}")
        return None, None, None

def predict_waste_item(model, label_encoder, category_names, image):
    """Predict specific waste item"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            category_name = category_names[idx]
            confidence = prediction[0][idx]
            top_3_predictions.append({
                'category': category_name,
                'confidence': float(confidence),
                'confidence_percent': float(confidence * 100)
            })
        
        # Get the top prediction
        top_prediction = top_3_predictions[0]
        
        # Get recycling information
        recycling_info = DETAILED_CLASSIFICATION.get(top_prediction['category'], {
            'category': 'unknown',
            'type': 'Unknown',
            'reason': 'Category not found in database',
            'disposal': 'Check local recycling guidelines'
        })
        
        return {
            'top_prediction': top_prediction,
            'top_3_predictions': top_3_predictions,
            'recycling_info': recycling_info,
            'detailed_guidance': get_recycling_guidance(top_prediction['category'])
        }
        
    except Exception as e:
        return {"error": str(e)}

def format_category_name(category_name):
    """Format category name for display"""
    return category_name.replace('_', ' ').title()

def main():
    if len(sys.argv) != 2:
        print("Usage: python enhanced_inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    # Load model
    model, label_encoder, category_names = load_model()
    
    if model is None:
        print("Model not available. Please train the model first:")
        print("python train.py")
        sys.exit(1)
    
    try:
        # Load and predict image
        image = Image.open(image_path)
        result = predict_waste_item(model, label_encoder, category_names, image)
        
        if "error" not in result:
            print(f"\n🖼️  Image: {image_path}")
            print("=" * 60)
            
            # Top prediction
            top = result['top_prediction']
            print(f"🎯 **IDENTIFIED ITEM:** {format_category_name(top['category'])}")
            print(f"📊 **Confidence:** {top['confidence_percent']:.2f}%")
            print(f"🔢 **Raw Score:** {top['confidence']:.4f}")
            
            # Recycling information
            recycling = result['recycling_info']
            if recycling['category'] == 'recyclable':
                print(f"♻️  **CLASSIFICATION:** RECYCLABLE")
            elif recycling['category'] == 'non_recyclable':
                print(f"🚫 **CLASSIFICATION:** NON-RECYCLABLE")
            else:
                print(f"❓ **CLASSIFICATION:** UNKNOWN")
            
            print(f"📋 **Type:** {recycling['type']}")
            print(f"💡 **Why:** {recycling['reason']}")
            print(f"🗑️  **How to dispose:** {recycling['disposal']}")
            
            # Top 3 predictions
            print(f"\n🏆 **TOP 3 PREDICTIONS:**")
            for i, pred in enumerate(result['top_3_predictions'], 1):
                print(f"  {i}. {format_category_name(pred['category'])} ({pred['confidence_percent']:.1f}%)")
            
            # Detailed guidance
            print(f"\n📖 **DETAILED GUIDANCE:**")
            print(result['detailed_guidance'])
            
        else:
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
