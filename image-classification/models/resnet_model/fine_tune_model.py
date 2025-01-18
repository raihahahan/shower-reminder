import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import platform

from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback

# Add the scripts directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.dataset_preprocessing import load_and_preprocess_dataset
import evaluate

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.5):  # Increased gamma for harder examples
        super().__init__()
        self.gamma = gamma
    
    def forward(self, inputs, targets, weight=None):
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def get_class_weights(dataset):
    """
    Computes class weights based on the dataset.
    """
    labels = np.array([example['label'] for example in dataset['train']])
    unique_classes = np.unique(labels)
    
    # Calculate class distribution
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Adjust weights more aggressively for the underrepresented class
    weights = np.array([
        (total_samples / (2 * class_counts[0])),  # not_showerhead
        (total_samples / (2 * class_counts[1]))   # showerhead
    ])
    
    # Normalize weights
    weights = weights / np.sum(weights)
    return dict(enumerate(weights))

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, continue_training=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_loss = FocalLoss(gamma=2.5)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            if len(pixel_values.shape) == 3:
                pixel_values = pixel_values.unsqueeze(0)
            outputs = model(pixel_values=pixel_values)
        else:
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        
        logits = outputs.logits
        
        # Compute loss with label smoothing
        if labels is not None:
            smoothing = 0.1
            labels_one_hot = F.one_hot(labels, num_classes=2).float()
            labels_smooth = (1.0 - smoothing) * labels_one_hot + smoothing / 2
            
            weights_tensor = torch.tensor(
                [self.class_weights[0], self.class_weights[1]], 
                device=labels.device,
                dtype=torch.float
            )
            
            ce_loss = F.cross_entropy(logits, labels, weight=weights_tensor)
            focal_loss = self.focal_loss(logits, labels, weight=weights_tensor)
            loss = 0.5 * ce_loss + 0.5 * focal_loss
            
            return (loss, outputs) if return_outputs else loss


def fine_tune_resnet(output_dir="models/fine_tuned_resnet", continue_training=False):
    """
    Fine-tunes a pre-trained ResNet model for binary classification.
    """
    # Load and preprocess the dataset
    dataset = load_and_preprocess_dataset(data_dir="dataset")
    
    # Compute class weights
    weights = get_class_weights(dataset)
    print(f"Class weights: {weights}")

    # Initialize the pre-trained model
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=2,
        id2label={0: "not_showerhead", 1: "showerhead"},
        label2id={"not_showerhead": 0, "showerhead": 1},
        ignore_mismatched_sizes=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu")
    use_bf16 = True if device.type == "mps" else False

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,  # Increased epochs for better learning
        weight_decay=0.01,
        logging_steps=100,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        bf16=torch.backends.mps.is_available(),  # Use mixed precision on MPS
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Using F1 score to balance precision and recall
        greater_is_better=True,
        save_total_limit=1,
        dataloader_num_workers=4,
        resume_from_checkpoint=continue_training,
    )

    # Initialize the trainer
    trainer = CustomTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        continue_training=continue_training,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    )

    # Train the model
    trainer.train()

    # Save the model
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)

def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1-score.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    accuracy = (predictions == labels).mean()
    
    true_pos = ((predictions == 1) & (labels == 1)).sum()
    false_pos = ((predictions == 1) & (labels == 0)).sum()
    false_neg = ((predictions == 0) & (labels == 1)).sum()
    
    precision = float(true_pos) / float(true_pos + false_pos + 1e-7)
    recall = float(true_pos) / float(true_pos + false_neg + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    accuracy = (predictions == labels).mean()
    
    true_pos = ((predictions == 1) & (labels == 1)).sum()
    false_pos = ((predictions == 1) & (labels == 0)).sum()
    false_neg = ((predictions == 0) & (labels == 1)).sum()
    
    precision = float(true_pos) / float(true_pos + false_pos + 1e-7)
    recall = float(true_pos) / float(true_pos + false_neg + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_training', action='store_true', 
                        help='Continue training from a saved checkpoint')
    args = parser.parse_args()
    
    fine_tune_resnet(output_dir="models/fine_tuned_resnet", continue_training=args.continue_training)
