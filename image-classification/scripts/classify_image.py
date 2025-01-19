import torch
from PIL import Image
import base64
import io
from transformers import ViTImageProcessor, ViTForImageClassification

class ImageClassifier:
    def __init__(self):
        # Load model and processor
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bytes):
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                
                # Get predicted class and confidence
                predicted_class_idx = logits.argmax(-1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                predicted_class = self.model.config.id2label[predicted_class_idx]
                
                # Check if it's something we want to classify as a showerhead
                is_bathroom_related = any(word in predicted_class.lower() 
                                        for word in ['shower', 'bathroom', 'plumbing', 'faucet'])
                
                return is_bathroom_related, confidence
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            return False, 0.0

# Initialize classifier once and reuse
classifier = ImageClassifier()
