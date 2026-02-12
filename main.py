import torch
from src.utils import recursive_unzip , TRAIN_DIR, TRAIN_CLEANED_DIR, TEST_DIR

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

recursive_unzip('denoising-dirty-documents.zip', extract_to='data/')