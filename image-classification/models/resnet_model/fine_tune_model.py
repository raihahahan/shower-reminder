import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import platform

from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import get_cosine_schedule_with_warmup

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
    labels = np.array([example['label'] for example in dataset['train']])
    class_counts = np.bincount(labels)
    
    # More aggressive weighting for minority class
    minority_weight = (class_counts.sum() / (2 * class_counts.min()))
    majority_weight = (class_counts.sum() / (2 * class_counts.max()))
    
    weights = np.array([majority_weight, minority_weight])
    if class_counts[0] > class_counts[1]:
        weights = weights[::-1]  # Reverse if necessary
        
    return dict(enumerate(weights))


class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, continue_training=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_loss = FocalLoss(gamma=2.5)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=int(num_training_steps * self.args.warmup_ratio),
                num_training_steps=num_training_steps
            )
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        try:
            labels = inputs.get("labels")
            pixel_values = inputs.get("pixel_values")
            
            # Ensure pixel_values has the correct shape
            if pixel_values is not None:
                if len(pixel_values.shape) == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                # Ensure pixel_values are contiguous in memory
                pixel_values = pixel_values.contiguous()
                outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            
            logits = outputs.logits
            
            if labels is not None:
                # Ensure labels are contiguous
                labels = labels.contiguous()
                
                # Label smoothing
                smoothing = 0.1
                num_classes = logits.size(-1)
                labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
                labels_smooth = (1.0 - smoothing) * labels_one_hot + smoothing / num_classes
                
                # Ensure weights tensor is on the correct device and contiguous
                weights_tensor = torch.tensor(
                    [self.class_weights[0], self.class_weights[1]], 
                    device=labels.device,
                    dtype=torch.float
                ).contiguous()
                
                # Compute losses
                ce_loss = F.cross_entropy(logits, labels, weight=weights_tensor)
                focal_loss = self.focal_loss(logits, labels, weight=weights_tensor)
                loss = 0.5 * ce_loss + 0.5 * focal_loss
                
                return (loss, outputs) if return_outputs else loss
                
        except Exception as e:
            print(f"Error in compute_loss: {e}")
            raise e


def fine_tune_convnext(output_dir="models/fine_tuned_convnext", continue_training=False):
    dataset_path = os.path.join(
        os.path.dirname(__file__), 
        "../../dataset" 
    )
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_and_preprocess_dataset(data_dir=dataset_path)
    
    weights = get_class_weights(dataset)
    print(f"Class weights: {weights}")

    # Use ConvNeXT-Tiny instead of Base
    model = AutoModelForImageClassification.from_pretrained(
        "facebook/convnext-tiny-224",  # Changed from base to tiny
        num_labels=2,
        id2label={0: "not_showerhead", 1: "showerhead"},
        label2id={"not_showerhead": 0, "showerhead": 1},
        ignore_mismatched_sizes=True,
    )
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
        model = model.to(device)

    # Adjusted training arguments for smaller model
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,  # Reduced batch size
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        weight_decay=0.01,
        logging_steps=50,
        warmup_ratio=0.1,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Using F1 score to balance precision and recall
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=2,  # Reduced workers
        bf16=False,  # Disabled mixed precision
        resume_from_checkpoint=continue_training,
        fp16=False,
        lr_scheduler_type="linear",  # Changed to linear scheduler
    )

    trainer = CustomTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        continue_training=continue_training,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
        ]
    )

    trainer.train()
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
    
    fine_tune_convnext(output_dir="models/fine_tuned_convnext", continue_training=args.continue_training)
