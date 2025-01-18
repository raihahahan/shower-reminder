from transformers import AutoModelForImageClassification

def load_saved_model():
    # Path to the saved model
    model_path = "models/resnet_model"

    # Load the model from the saved directory
    print(f"Loading model from: {model_path}")
    model = AutoModelForImageClassification.from_pretrained(model_path)
    print("Model loaded successfully!")

    return model

if __name__ == "__main__":
    model = load_saved_model()
