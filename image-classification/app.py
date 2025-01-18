# app.py
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoModelForImageClassification

app = Flask(__name__)

# Update model path to point to the correct directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fine_tuned_resnet')

def load_model(model_path=MODEL_PATH):
    print(f"Loading model from: {model_path}")
    try:
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label={0: "not_showerhead", 1: "showerhead"},
            label2id={"not_showerhead": 0, "showerhead": 1},
            local_files_only=True  # Add this to ensure it loads from local path
        )
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def get_transforms():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image_base64):
    try:
        # Decode base64 to image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Transform image
        transforms = get_transforms()
        image_tensor = transforms(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return predicted_class == 1, confidence
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return False, 0.0

# Load model globally with error handling
try:
    print("Attempting to load model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        is_showerhead, confidence = predict_image(model, data['image'])
        
        return jsonify({
            'is_showerhead': is_showerhead,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000, debug=True)  # Changed port to 8000
    except Exception as e:
        print(f"Failed to start server: {e}")