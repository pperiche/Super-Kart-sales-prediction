
from huggingface_hub import HfApi
import os

# Initialize API with token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload Streamlit app folder to Hugging Face Space
api.upload_folder(
    folder_path="SuperKart",   # root folder containing app.py, models, etc.
    repo_id="PratzPrathibha/superkart-sales-prediction-app",  # <-- your HF Space repo
    repo_type="space",  # IMPORTANT: this must be 'space'
    path_in_repo="",  # upload to root
)

print("Deployment to Hugging Face Space completed successfully")
