from huggingface_hub import snapshot_download, HfApi
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import time
import requests
from urllib3.exceptions import IncompleteRead
from typing import Optional, List


class WavCapsDownloader:
    AVAILABLE_DATASETS = {
        'freesound': 'FreeSound',
        'bbc': 'BBC_Sound_Effects',
        'soundbible': 'SoundBible',
        'audioset': 'AudioSet_SL'
    }

    def __init__(self,
                 output_dir: str = "./data/wavcaps",
                 max_retries: int = 3,
                 chunk_size: int = 8192
                 ):
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.api = HfApi()

    @classmethod
    def list_available_datasets(cls):
        """Print available datasets."""
        print("\nAvailable datasets:")
        for key, name in cls.AVAILABLE_DATASETS.items():
            print(f"- {key}: {name}")

    def download_dataset(self, datasets: Optional[List[str]] = None) -> Path:
        """
        Download selected WavCaps datasets with retry logic.
        
        Args:
            datasets: List of dataset keys to download ('freesound', 'bbc', 'soundbible', 'audioset').
                     If None, downloads all datasets.
        
        Returns:
            Path: Path to the downloaded dataset
        """
        if datasets:
            invalid_datasets = [
                d for d in datasets if d not in self.AVAILABLE_DATASETS]
            if invalid_datasets:
                raise ValueError(
                    f"Invalid dataset(s): {invalid_datasets}. Available options: {list(self.AVAILABLE_DATASETS.keys())}")

        print("Starting WavCaps dataset download...")
        print(f"Selected datasets: {datasets}")
        print(f"Output directory: {self.output_dir}")
        selected_patterns = []

        if datasets:
            # Create patterns for selected datasets
            for dataset in datasets:
                name = self.AVAILABLE_DATASETS[dataset]
                patterns = [
                    f"**/{name}/**",
                ]
                print(f"Patterns for dataset '{dataset}': {patterns}")
                print("Debug: Starting download with the following parameters:")
                print(
                    f"repo_id='cvssp/WavCaps', repo_type='dataset', local_dir='{self.output_dir}', resume_download=True, max_workers=4, allow_patterns={selected_patterns if datasets else None}")
                selected_patterns.extend(patterns)

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            for attempt in range(self.max_retries):
                try:
                    print(
                        f"Attempting to download dataset (attempt {attempt + 1})...")
                    print("Calling snapshot_download with the following parameters:")
                    print(
                        f"repo_id='cvssp/WavCaps', repo_type='dataset', local_dir='{self.output_dir}', resume_download=True, max_workers=4, allow_patterns={selected_patterns if datasets else None}")
                    dataset_dir = snapshot_download(
                        repo_id="cvssp/WavCaps",
                        repo_type="dataset",
                        local_dir=str(self.output_dir),
                        resume_download=True,
                        max_workers=4,
                        allow_patterns=selected_patterns if datasets else None
                    )
                    print(f"Dataset downloaded successfully to: {dataset_dir}")
                    return Path(dataset_dir)

                except Exception as e:
                    print(
                        f"Error during download (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                    print(
                        "Please check your internet connection and ensure the Hugging Face API is accessible.")
                    if attempt < self.max_retries - 1:
                        print(
                            f"\nError during download (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                        print("Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        raise

        except Exception as e:
            print(
                f"Failed to download dataset after {self.max_retries} attempts: {str(e)}")
            raise

    def extract_zip_files(self, dataset_dir: Path):
        """
        Extract all zip files in the dataset directory.
        
        Args:
            dataset_dir: Path to the directory containing zip files.
        """
        print("\nExtracting zip files...")
        for zip_path in dataset_dir.rglob("*.zip"):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    extract_path = zip_path.parent / zip_path.stem
                    zip_ref.extractall(extract_path)
                    print(f"Extracted {zip_path} to {extract_path}")
            except zipfile.BadZipFile:
                print(f"Failed to extract {zip_path}: Bad zip file")


def main():
    """Main function to download and extract WavCaps dataset."""
    downloader = WavCapsDownloader()

    # Show available datasets
    WavCapsDownloader.list_available_datasets()

    # Get user input for dataset selection
    print("\nEnter dataset keys to download (comma-separated) or press Enter for all:")
    print("Example: freesound,bbc")

    user_input = input("> ").strip()
    selected_datasets = [d.strip()
                         for d in user_input.split(',')] if user_input else None

    try:
        # Download selected datasets
        dataset_dir = downloader.download_dataset(selected_datasets)

        # Extract zip files
        downloader.extract_zip_files(dataset_dir)

        print("\nDataset processing completed successfully!")

        # Dynamically print ASCII directory structure for downloaded datasets
        print("\nASCII Directory Structure:")
        for dataset in selected_datasets or downloader.AVAILABLE_DATASETS.keys():
            dataset_name = downloader.AVAILABLE_DATASETS[dataset]
            print(f"WavCaps Dataset/")
            print(f"├── {dataset_name}/")
            print(f"│   ├── json_files/")
            print(f"│   └── Zip_files/")
        print("\nDirectory structure:")
        for item in dataset_dir.rglob("*"):
            if item.is_file():
                print(f"  {item.relative_to(dataset_dir)}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
