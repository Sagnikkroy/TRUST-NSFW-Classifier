import os
from datasets import load_dataset
from tqdm import tqdm

# 1. Define Paths
# Point to the root of your cache, not the deep snapshot folder
CACHE_DIR = "E:/huggingface_ds"
SAVE_PATH = "E:/nsfw_data"

print("Loading dataset from cache...")
dataset = load_dataset("deepghs/nsfw_detect", cache_dir=CACHE_DIR)

# 2. Split: 80% Train, 10% Val, 10% Test
print("Shuffling and splitting data...")
train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)

# Create a dictionary to make looping easier for the progress bar
splits = {
    "train": train_test["train"],
    "val": test_val["train"],
    "test": test_val["test"]
}

# 3. Save with Progress Bar
print(f"Saving splits to {SAVE_PATH}...")
for name, data in tqdm(splits.items(), desc="Saving Dataset Splits"):
    # This saves each folder (train, val, test) to your E: drive
    data.save_to_disk(os.path.join(SAVE_PATH, name))

print(f"\n noice {SAVE_PATH}")