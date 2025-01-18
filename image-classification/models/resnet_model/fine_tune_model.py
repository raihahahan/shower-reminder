import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.dataset_preprocessing import load_and_preprocess_dataset
import evaluate

def get_class_weights(dataset):
    labels = np.array([example['label'] for example in dataset['train']])
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    return dict(enumerate(class_weights))

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        
        weights_tensor = torch.tensor(
            [self.class_weights[0], self.class_weights[1]], 
            device=labels.device,
            dtype=torch.float
        )
        
        loss = F.cross_entropy(logits, labels, weight=weights_tensor)
        
        return (loss, outputs) if return_outputs else loss

def fine_tune_resnet(output_dir="models/fine_tuned_resnet"):
    # Load the dataset
    dataset = load_and_preprocess_dataset(data_dir="dataset")
    
    # Calculate class weights
    weights = get_class_weights(dataset)
    print(f"Class weights: {weights}")

    # Load the pre-trained ResNet-50 model
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=2,
        id2label={0: "not_showerhead", 1: "showerhead"},
        label2id={"not_showerhead": 0, "showerhead": 1},
        ignore_mismatched_sizes=True,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        logging_dir="./logs",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.05,
        save_total_limit=2,
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        use_mps_device=True
    )

    # Define evaluation metric
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Initialize the CustomTrainer
    trainer = CustomTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
        ]
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to: {output_dir}")

if __name__ == "__main__":
    fine_tune_resnet()