import os
import torch
from datasets import load_from_disk
from transformers import pipeline
from tqdm import tqdm
from PIL import Image

# 1. SETUP
MODEL_PATH = "./final_nsfw_classifier"
DATA_PATH = "E:/nsfw_data/test"  # Use the unseen test data
OUTPUT_DIR = "./false_positives"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model and test dataset...")
classifier = pipeline("image-classification", model=MODEL_PATH, device=0)
test_ds = load_from_disk(DATA_PATH)

results = []

# 2. ANALYSIS LOOP
print("Scanning test set for False Positives...")
# We check the first 500 images of the test set for a quick audit
for i in tqdm(range(len(test_ds))):
    example = test_ds[i]
    true_label_idx = example['label']
    true_label_name = test_ds.features['label'].names[true_label_idx]
    
    # Run prediction
    img = example['image'].convert("RGB")
    prediction = classifier(img)[0] # Get top prediction
    pred_label = prediction['label']
    pred_score = prediction['score']

    # 3. IDENTIFY FALSE POSITIVES
    # Definition: Actual is 'neutral', but predicted is 'porn', 'hentai', or 'sexy'
    nsfw_labels = ['porn', 'hentai', 'sexy']
    
    if true_label_name == 'neutral' and pred_label in nsfw_labels:
        # Save the info
        results.append({
            'index': i,
            'true': true_label_name,
            'pred': pred_label,
            'score': pred_score
        })
        
        # Save the actual image so you can look at it
        img.save(f"{OUTPUT_DIR}/fp_{i}_{pred_label}_{pred_score:.2f}.jpg")

# 4. SUMMARY
print(f"\n--- Analysis Complete ---")
print(f"Total False Positives found: {len(results)}")
print(f"Images saved to: {OUTPUT_DIR}")