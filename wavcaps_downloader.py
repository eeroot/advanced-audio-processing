from huggingface_hub import snapshot_download, HfApi
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import time
import requests
from urllib3.exceptions import IncompleteRead
from typing import Optional

class WavCapsDownloader:
    def __init__(self, output_dir: str = "./wavcaps_dataset", max_retries: int = 3, chunk_size: int = 8192):
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.api = HfApi()

    def download_file_with_resume(self, url: str, dest_path: Path, file_size: Optional[int] = None) -> bool:
        """
        Download a file with resume capability and retry logic.
        
        Args:
            url (str): URL to download from
            dest_path (Path): Destination path
            file_size (int, optional): Expected file size
            
        Returns:
            bool: True if download was successful
        """
        headers = {}
        mode = 'wb'
        
        # Resume download if file exists
        if dest_path.exists():
            current_size = dest_path.stat().st_size
            if file_size and current_size >= file_size:
                print(f"File {dest_path.name} already completely downloaded")
                return True
            headers['Range'] = f'bytes={current_size}-'
            mode = 'ab'
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=headers, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(dest_path, mode) as f:
                    with tqdm(
                        total=total_size,
                        initial=dest_path.stat().st_size if dest_path.exists() else 0,
                        unit='iB',
                        unit_scale=True,
                        desc=dest_path.name
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                return True
                
            except (IncompleteRead, requests.exceptions.RequestException) as e:
                print(f"\nError downloading {dest_path.name} (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                time.sleep(5)  # Wait before retrying
                
        return False

    def download_dataset(self) -> Path:
        """
        Download the WavCaps dataset with retry logic.
        
        Returns:
            Path: Path to the downloaded dataset
        """
        print("Starting WavCaps dataset download...")
        
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Download using snapshot_download with retry logic
            for attempt in range(self.max_retries):
                try:
                    dataset_dir = snapshot_download(
                        repo_id="cvssp/WavCaps",
                        repo_type="dataset",
                        local_dir=str(self.output_dir),
                        resume_download=True,
                        max_workers=4  # Limit concurrent downloads
                    )
                    print(f"Dataset downloaded successfully to: {dataset_dir}")
                    return Path(dataset_dir)
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        print(f"\nError during download (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                        print("Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        raise
                        
        except Exception as e:
            print(f"Failed to download dataset after {self.max_retries} attempts: {str(e)}")
            raise

    def extract_zip_files(self, directory: Path) -> None:
        """
        Extract zip files with error handling and resume capability.
        """
        print("\nExtracting zip files...")
        
        try:
            zip_files = list(directory.rglob("*.zip"))
            if not zip_files:
                print("No zip files found.")
                return

            for zip_path in zip_files:
                extract_dir = zip_path.with_suffix('')
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"\nProcessing: {zip_path.name}")
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        files = zip_ref.namelist()
                        
                        # Skip already extracted files
                        files_to_extract = [
                            f for f in files 
                            if not (extract_dir / f).exists()
                        ]
                        
                        if not files_to_extract:
                            print(f"All files already extracted in: {extract_dir}")
                            continue
                            
                        with tqdm(total=len(files_to_extract), desc="Extracting") as pbar:
                            for file in files_to_extract:
                                zip_ref.extract(file, extract_dir)
                                pbar.update(1)
                                
                except zipfile.BadZipFile as e:
                    print(f"Error: Corrupted zip file {zip_path.name} - {e}")
                    # Optionally delete corrupted file and retry download
                    
        except Exception as e:
            print(f"Error during extraction: {e}")
            raise

def main():
    """Main function to download and extract WavCaps dataset."""
    downloader = WavCapsDownloader()
    
    try:
        # Download dataset
        dataset_dir = downloader.download_dataset()
        
        # Extract zip files
        downloader.extract_zip_files(dataset_dir)
        
        print("\nDataset processing completed successfully!")
        
        # Print directory structure
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