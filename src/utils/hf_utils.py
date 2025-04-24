from huggingface_hub import HfApi, HfFolder

def save(file_paths, repo_id, commit_message):
    api = HfApi()
    token = HfFolder.get_token()  # Ensure you have logged in using `huggingface-cli login`

        # Upload the files
    for file_path in file_paths:
        api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,  # Keep relative path in repo
                repo_id=repo_id,
                repo_type="model",  # Change to "dataset" if uploading to a dataset repo
                commit_message=commit_message,
            )