from transformers import AutoModelForImageClassification

def save_resnet_model():
    # Specify the model name
    model_name = "microsoft/resnet-50"

    # Load the pre-trained model from Hugging Face
    print(f"Loading pre-trained model: {model_name}")
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # Save the model weights to the specified directory
    save_path = "models/resnet_model"
    print(f"Saving model weights to: {save_path}")
    model.save_pretrained(save_path)

    print("Model saved successfully!")

if __name__ == "__main__":
    save_resnet_model()
