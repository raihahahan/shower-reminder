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
    
    train_transforms = Compose([
        Resize((256, 256)),  # Larger initial size
        RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.3),
        RandomRotation(45),
        RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
        RandomPerspective(distortion_scale=0.5, p=0.5),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        RandomGrayscale(p=0.1),
        GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ToTensor(),
        normalize
    ])
    
    val_transforms = Compose([
        Resize((224, 224)),
        CenterCrop(224),
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