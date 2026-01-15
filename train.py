import os
import torch
import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor, 
    Trainer, 
    TrainingArguments,
    DefaultDataCollator,
    TrainerCallback
)

# 1. SETUP PATHS
DATA_PATH = "E:/nsfw_data"
CHECKPOINT_DIR = "E:/nsfw_model_checkpoints"
FINAL_MODEL_DIR = "./final_nsfw_classifier"

# 2. LOAD DATA
print("--- Loading Prepared Data from E: Drive ---")
train_ds = load_from_disk(os.path.join(DATA_PATH, "train"))
val_ds = load_from_disk(os.path.join(DATA_PATH, "val"))

# Get labels for the 5 categories
labels = train_ds.features['label'].names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# 3. PREPROCESSING (SMART TRANSFORM)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def transform(example_batch):
    # DYNAMIC KEY DETECTION: Checks if the column is 'image' or 'img'
    img_key = 'image' if 'image' in example_batch else 'img'
    
    # Process images and prepare labels
    inputs = processor([x.convert("RGB") for x in example_batch[img_key]], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

train_ds.set_transform(transform)
val_ds.set_transform(transform)

# 4. EVALUATION METRICS
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. INITIALIZE MODEL
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# 6. TRAINING CONFIG (Optimized for Accuracy & Latest Library Version)
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="epoch",           # NEW: Replaced evaluation_strategy
    save_strategy="epoch",           
    learning_rate=2e-5,              
    per_device_train_batch_size=8,   # Smaller batch size to prevent OOM errors
    per_device_eval_batch_size=8,
    num_train_epochs=5,              
    weight_decay=0.01,
    load_best_model_at_end=True,     # Keeps the most accurate version
    metric_for_best_model="accuracy",
    logging_steps=10,                
    save_total_limit=2,              
    fp16=True if torch.cuda.is_available() else False, # GPU acceleration
    remove_unused_columns=False,
)

# 7. INITIALIZE TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=DefaultDataCollator(),
)

# 8. RUN TRAINING (WITH AUTOMATIC RESUME)
print("\n--- Starting Live Training ---")
resume_from = None
# Automatically check if a checkpoint exists on E: drive
if os.path.exists(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
    resume_from = True
    print(f"Resuming from latest checkpoint in {CHECKPOINT_DIR}...")

trainer.train(resume_from_checkpoint=resume_from)

# 9. SAVE FINAL RESULT
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print(f"\n SUCCESS:model saved to {FINAL_MODEL_DIR}")