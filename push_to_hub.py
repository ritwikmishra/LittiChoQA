# import os
# from huggingface_hub import HfApi, upload_folder

# def push_adapter_to_huggingface(
#     local_adapter_path: str,
#     repo_name: str,
#     token: str,
#     private: bool = True,
# ):
#     """
#     Push only the necessary fine-tuned adapter files to Hugging Face Hub.
#     """
#     repo_name = "ritwikm/" + repo_name
#     api = HfApi()

#     # 1️⃣ Create repo if it doesn't exist
#     print(f"Checking or creating repo: {repo_name}")
#     api.create_repo(repo_id=repo_name, private=private, exist_ok=True, token=token)

#     # 2️⃣ List of files to include
#     required_files = [
#         "adapter_model.safetensors",
#         "adapter_config.json",
#         "tokenizer_config.json",
#         "special_tokens_map.json",
#     ]

#     # Filter only existing files
#     files_to_upload = [f for f in required_files if os.path.exists(os.path.join(local_adapter_path, f))]
    
#     if not files_to_upload:
#         print("No required adapter files found in the folder. Aborting upload.")
#         return

#     print(f"Uploading the following files: {files_to_upload}")

#     # 3️⃣ Upload selected files
#     upload_folder(
#         folder_path=local_adapter_path,
#         repo_id=repo_name,
#         token=token,
#         repo_type="model",
#         ignore_patterns=[f for f in os.listdir(local_adapter_path) if f not in files_to_upload]
#     )

#     print(f"Successfully pushed adapter to https://huggingface.co/{repo_name}")


# # Usage
# local_adapter_path = "/media/data_dump/Ritwik/git/mqna/data/jsonData2/out/gn3/checkpoint-23316"
# repo_name = "gn3"
# token = "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"

# push_adapter_to_huggingface(local_adapter_path, repo_name, token)

from huggingface_hub import HfApi, upload_file

def create_and_upload_dataset_to_huggingface(
    username: str,
    repo_name: str,
    token: str,
    local_file_path: str,
    private: bool = False,
    license_name: str = "cc-by-nc-4.0"
):
    """
    Create a Hugging Face dataset repository, upload a JSON file,
    and set license metadata (e.g., CC-BY-NC 4.0 for research-only use).
    """
    api = HfApi()
    full_repo_name = f"{username}/{repo_name}"

    print(f"🔍 Checking or creating dataset repo: {full_repo_name}")

    try:
        # Step 1: Create (or reuse) the dataset repo
        api.create_repo(
            repo_id=full_repo_name,
            repo_type="dataset",
            token=token,
            private=private,
            exist_ok=True
        )
        print(f"✅ Dataset repository ready at: https://huggingface.co/datasets/{full_repo_name}")

        # Step 2: Upload your local dataset file
        print(f"⬆️ Uploading file from: {local_file_path}")
        upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo="MuNfQuAD_v2.json",
            repo_id=full_repo_name,
            repo_type="dataset",
            token=token
        )
        print(f"✅ File uploaded successfully to the dataset hub.")

        # Step 3: Add license and metadata
        api.update_repo_metadata(
            repo_id=full_repo_name,
            repo_type="dataset",
            metadata={
                "license": license_name,
                "language": ["en"],
                "annotations_creators": ["expert-generated"],
                "task_categories": ["question-answering"]
            },
            token=token
        )
        print(f"📜 License set to: {license_name}")

        print(f"🎉 Dataset successfully uploaded to: https://huggingface.co/datasets/{full_repo_name}")

    except Exception as e:
        print(f"❌ Failed to upload dataset: {e}")


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    username = "ritwikm"  # Your HF username
    repo_name = "MuNfQuAD-v2"  # Dataset repo name
    token = "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"  # Your Hugging Face token
    local_file_path = "/media/data_dump/Ritwik/git/mqna/data/dataset/MuNfQuAD/MuNfQuAD_v2.json"

    create_and_upload_dataset_to_huggingface(
        username=username,
        repo_name=repo_name,
        token=token,
        local_file_path=local_file_path,
        private=False,
        license_name="cc-by-nc-4.0"
    )
