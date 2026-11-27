"""Copy the processed data into the Space folder and upload it to Hugging Face Spaces."""
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = os.environ.get("SPACE_ID", "devinmcintosha/boj-policy-tone")

dest = ROOT / "space" / "data"
dest.mkdir(exist_ok=True)
for name in ("statements.csv", "sentences.csv", "summary.json"):
    shutil.copy(ROOT / "data" / "processed" / name, dest / name)

api = HfApi(token=os.environ["HF_TOKEN"])
if not api.repo_exists(SPACE_ID, repo_type="space"):
    api.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio")
api.upload_folder(folder_path=str(ROOT / "space"), repo_id=SPACE_ID, repo_type="space",
                  commit_message="Update data and app")
print(f"published https://huggingface.co/spaces/{SPACE_ID}")
