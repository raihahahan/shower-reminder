from datasets import load_dataset, DatasetDict
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter,
    RandomAffine, RandomPerspective, RandomResizedCrop, RandomGrayscale,
    GaussianBlur, CenterCrop
)

def load_and_preprocess_dataset(data_dir="dataset"):
    dataset = load_dataset("imagefolder", data_dir=data_dir)
    
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Simpler transforms
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        normalize
    ])

    def preprocess_images(examples):
        if "image" not in examples:
            raise ValueError("Expected 'image' key in examples but not found.")

        examples["pixel_values"] = [
            transforms(image.convert("RGB")).numpy() 
            for image in examples["image"]
        ]
        return examples

    processed_dataset = dataset.map(
        preprocess_images,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Processing dataset"
    )

    processed_dataset.set_format(type="torch", columns=["pixel_values", "label"])
    return processed_dataset

    def apply_extra_augmentations(image):
        # Extra augmentations for minority class
        extra_transforms = Compose([
            RandomHorizontalFlip(p=0.8),
            RandomRotation(45),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        ])
        return extra_transforms(image)

    # Apply preprocessing to train and validation datasets separately
    # Apply preprocessing to train and validation datasets separately
    train_dataset = dataset["train"].map(
        lambda examples: preprocess_images(examples, train_transforms),
        batched=True,
        remove_columns=dataset["train"].column_names,  # Remove all original columns
        desc="Processing training dataset"
    )
    
    val_dataset = dataset["validation"].map(
        lambda examples: preprocess_images(examples, val_transforms),
        batched=True,
        remove_columns=dataset["validation"].column_names,  # Remove all original columns
        desc="Processing validation dataset"
    )

    # Set the format to PyTorch tensors
    train_dataset.set_format(type="torch", columns=["pixel_values", "label"])
    val_dataset.set_format(type="torch", columns=["pixel_values", "label"])

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })