import os
from datasets import load_dataset

# 1. Define your path
cache_path = "E:/huggingface_ds"

# 2. Ensure the directory exists
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

try:
    # Use 'token' instead of 'use_auth_token'
    dataset = load_dataset(
        "deepghs/nsfw_detect",
        token="", 
        cache_dir=cache_path
    )
    
    print("Success! Dataset loaded")
    labels = dataset['train'].features['label'].names
    print(f"Labels: {labels}")

except Exception as e:
    print(f"Error: {e}")
