import os
from datasets import load_dataset, DownloadConfig

# Create the directory if it doesn't exist
cache_path = "E:/huggingface_ds"
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

# Force the environment variables within the script as a backup
os.environ["HF_HOME"] = cache_path

dataset = load_dataset(
    "deepghs/nsfw_detect", 
    token="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", 
    cache_dir=cache_path,
    download_config=DownloadConfig(cache_dir=cache_path)
)

print("Download successful!")
print(f"Labels found: {dataset['train'].features['label'].names}")