"""
Downloads model artifacts from Hugging Face Hub at container startup.
Keeps binary files out of the git repository.
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID       = 'mansoorsk/matchlens-artifacts'
ARTIFACTS_DIR = 'artifacts'
FILES         = [
    'svm_classifier.joblib',
    'tfidf_vectorizer.joblib',
    'id2label.pkl',
    'label2id.pkl'
]

def download_artifacts():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    for filename in FILES:
        dest = f'{ARTIFACTS_DIR}/{filename}'

        if os.path.exists(dest):
            print(f'Already exists: {filename}')
            continue

        print(f'Downloading {filename}...')
        hf_hub_download(
            repo_id  = REPO_ID,
            filename = filename,
            local_dir= ARTIFACTS_DIR,
            token    = os.environ.get('HF_TOKEN')
        )
        print(f'Downloaded: {filename}')

    print('All artifacts ready.')

if __name__ == '__main__':
    download_artifacts()
