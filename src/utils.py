"""
Helper functions: image loading, paths, logging.
"""

import os
import zipfile
from pathlib import Path
from typing import Union


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TRAIN_CLEANED_DIR = DATA_DIR / "train_cleaned"
TEST_DIR = DATA_DIR / "test"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sampleSubmission.csv"


def recursive_unzip(zip_path: Union[str, Path], extract_to: Union[str, Path] = None, remove_after: bool = False) -> None:
    """
    Recursively extract a zip file and all nested zip files within it.
    
    Args:
        zip_path: Path to the zip file to extract
        extract_to: Directory to extract to. If None, extracts to the same directory as zip_path
        remove_after: If True, removes zip files after extraction
    
    Returns:
        None
    """
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"File is not a valid zip file: {zip_path}")
    
    # Determine extraction directory
    if extract_to is None:
        extract_to = zip_path.parent / zip_path.stem
    else:
        extract_to = Path(extract_to)
    
    # Create extraction directory if it doesn't exist
    extract_to.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_to}")
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Remove the zip file if requested
    if remove_after:
        zip_path.unlink()
        print(f"Removed {zip_path}")
    
    # Find all zip files in the extracted directory and recursively extract them
    for item in extract_to.rglob('*.zip'):
        if item.is_file() and zipfile.is_zipfile(item):
            # Recursively extract nested zip files
            nested_extract_to = item.parent / item.stem
            recursive_unzip(item, nested_extract_to, remove_after=remove_after)
