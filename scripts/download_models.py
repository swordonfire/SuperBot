import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(repo_id: str, model_file: str, local_dir: str):
    """Download any GGUF model from Hugging Face Hub."""
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    # Download only the specified GGUF file
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=model_file,  # e.g., "*.Q4_K_M.gguf"
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f'Model saved to {local_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download GGUF models from Hugging Face')
    parser.add_argument(
        '--repo_id',
        type=str,
        required=True,
        help="HF repo ID (e.g., 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF')",
    )
    parser.add_argument(
        '--model_file',
        type=str,
        required=True,
        help="Exact GGUF filename or wildcard (e.g., '*Q4_K_M.gguf')",
    )
    parser.add_argument(
        '--local_dir', type=str, default='data/models', help='Local directory to save the model'
    )
    args = parser.parse_args()

    download_model(args.repo_id, args.model_file, args.local_dir)
