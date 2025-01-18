from datasets import load_dataset, DatasetDict
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomRotation, ColorJitter
)

def load_and_preprocess_dataset(data_dir="dataset"):
    # Load dataset from directory
    dataset = load_dataset("imagefolder", data_dir=data_dir)

    # Define preprocessing transformations
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Augmentations for training
    train_transforms = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.3),  
    RandomRotation(30),         
    RandomResizedCrop(224, scale=(0.8, 1.0)),  
    ColorJitter(
        brightness=0.3, 
        contrast=0.3, 
        saturation=0.3, 
        hue=0.1
    ),
    RandomGrayscale(p=0.1),    
    RandomAffine(degrees=0, translate=(0.1, 0.1)),  
    ToTensor(),
    normalize
    ])
    
    # Standard preprocessing for validation
    val_transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        normalize
    ])

    def preprocess_images(examples, transforms):
        if "image" not in examples:
            print(f"Batch keys: {examples.keys()}")
            raise ValueError("Expected 'image' key in examples but not found.")

        pixel_values = []
        for image in examples["image"]:
            try:
                transformed_image = transforms(image.convert("RGB"))
                pixel_values.append(transformed_image.numpy())
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
                
        examples["pixel_values"] = pixel_values
        return examples

    # Apply preprocessing to train and validation datasets separately
    train_dataset = dataset["train"].map(
        lambda examples: preprocess_images(examples, train_transforms),
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != "label"]
    )
    
    val_dataset = dataset["validation"].map(
        lambda examples: preprocess_images(examples, val_transforms),
        batched=True,
        remove_columns=[col for col in dataset["validation"].column_names if col != "label"]
    )

    # Set the format to PyTorch tensors
    train_dataset.set_format(type="torch", columns=["pixel_values", "label"])
    val_dataset.set_format(type="torch", columns=["pixel_values", "label"])

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })