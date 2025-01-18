import sys
import os

from transformers import AutoImageProcessor
from PIL import Image
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.resnet_model.load_model import load_saved_model

def classify_image(image_path):
    # Load model and feature extractor
    model = load_saved_model()
    feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=img, return_tensors="pt")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=3)

    # Decode labels
    labels = model.config.id2label
    predictions = [
        {"label": labels[idx.item()], "score": prob.item()}
        for idx, prob in zip(top_indices[0], top_probs[0])
    ]
    return predictions

if __name__ == "__main__":
    image_path = "data/images/showerhead.jpg"  # Replace with your test image
    result = classify_image(image_path)
    for pred in result:
        print(f"Label: {pred['label']}, Confidence: {pred['score']:.2f}")
