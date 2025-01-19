import torch
from PIL import Image
import io
import base64
from transformers import CLIPProcessor, CLIPModel

class ImageClassifier:
    def __init__(self):
        print("Initializing CLIP model...")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()
        print("Model initialized successfully")

    def predict(self, image_base64):
        try:
            # Decode base64 to image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Define custom class labels
            class_labels = ["showerhead", "not showerhead"]
            
            # Prepare inputs
            inputs = self.processor(
                text=class_labels, images=image, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # Shape: [1, num_classes]
                probabilities = logits_per_image.softmax(dim=1)
                
                # Get prediction
                predicted_idx = probabilities.argmax().item()
                confidence = probabilities[0][predicted_idx].item()
                predicted_class = class_labels[predicted_idx]
                
                is_showerhead = predicted_class.lower() == "showerhead"
                
                # If not a showerhead, include the predicted label
                if not is_showerhead:
                    ai_label = predicted_class  # Here, predicted_class will be "not showerhead"
                else:
                    ai_label = "showerhead"
                
                # Return showerhead flag, confidence, and AI's predicted label
                return is_showerhead, confidence, ai_label

        except Exception as e:
            print(f"Error in prediction: {e}")
            return "error", 0.0
