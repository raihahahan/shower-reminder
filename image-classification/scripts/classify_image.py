from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

def classify_image(image_path, model_dir="models/fine_tuned_resnet"):
    # Load the fine-tuned model and feature extractor
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=img, return_tensors="pt")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = probs.argmax(dim=-1).item()

    # Map predictions to labels
    id2label = {0: "not_showerhead", 1: "showerhead"}
    return id2label[predicted_class], probs[0][predicted_class].item()

if __name__ == "__main__":
    image_path = "dataset/val/showerhead/img1.jpg"  # Replace with your image path
    label, confidence = classify_image(image_path)
    print(f"Predicted: {label} with confidence {confidence:.2f}")
