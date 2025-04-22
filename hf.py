from huggingface_hub import HfApi, HfFolder

# Define the file paths and repository details
file_paths = [
    "data/checkpoints/cbow.82326lines.256embeddings.5minfreq.5epochs.pth",
    "data/checkpoints/cbow.500lines.256embeddings.0minfreq.5epochs.pth",
    "data/processed/ms_marco_tkn_word_to_ids_500_lines_minfreq_none.json",
    "data/processed/ms_marco_training_data_500_lines_minfreq_0.json",
]

repo_id = "andreadellacorte/ml-institute-week-2-two-towers"  # Replace with your Hugging Face repo
commit_message = "Upload checkpoint files"

# Authenticate with Hugging Face
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
