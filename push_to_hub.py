import os
from huggingface_hub import HfApi, upload_folder

def push_adapter_to_huggingface(
    local_adapter_path: str,
    repo_name: str,
    token: str,
    private: bool = True,
):
    """
    Push only the necessary fine-tuned adapter files to Hugging Face Hub.
    """
    repo_name = "ritwikm/" + repo_name
    api = HfApi()

    # 1️⃣ Create repo if it doesn't exist
    print(f"Checking or creating repo: {repo_name}")
    api.create_repo(repo_id=repo_name, private=private, exist_ok=True, token=token)

    # 2️⃣ List of files to include
    required_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    # Filter only existing files
    files_to_upload = [f for f in required_files if os.path.exists(os.path.join(local_adapter_path, f))]
    
    if not files_to_upload:
        print("No required adapter files found in the folder. Aborting upload.")
        return

    print(f"Uploading the following files: {files_to_upload}")

    # 3️⃣ Upload selected files
    upload_folder(
        folder_path=local_adapter_path,
        repo_id=repo_name,
        token=token,
        repo_type="model",
        ignore_patterns=[f for f in os.listdir(local_adapter_path) if f not in files_to_upload]
    )

    print(f"Successfully pushed adapter to https://huggingface.co/{repo_name}")


# Usage
local_adapter_path = "/media/data_dump/Ritwik/git/mqna/data/jsonData2/out/gn3/checkpoint-23316"
repo_name = "gn3"
token = "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"

push_adapter_to_huggingface(local_adapter_path, repo_name, token)

 