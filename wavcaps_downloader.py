
from huggingface_hub import snapshot_download
import os
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_wavcaps(output_dir: str = "./wavcaps_dataset") -> Path:
    """
    Download the WavCaps dataset from Hugging Face.
    
    Args:
        output_dir (str): Directory to store the dataset
        
    Returns:
        Path: Path to the downloaded dataset
    """
    print("Downloading WavCaps dataset...")
    try:
        dataset_dir = snapshot_download(
            repo_id="cvssp/WavCaps",
            repo_type="dataset",
            local_dir=output_dir
        )
        print(f"Dataset downloaded successfully to: {dataset_dir}")
        return Path(dataset_dir)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def extract_zip_files(directory: Path) -> None:
    """
    Extract all zip files in the directory and its subdirectories.
    
    Args:
        directory (Path): Directory containing zip files
    """
    print("\nExtracting zip files...")
    try:
        # Find all zip files
        zip_files = list(directory.rglob("*.zip"))
        if not zip_files:
            print("No zip files found.")
            return

        # Extract each zip file with progress bar
        for zip_path in zip_files:
            # Create extraction directory (same name as zip without extension)
            extract_dir = zip_path.with_suffix('')
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nExtracting: {zip_path.name}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files for progress bar
                files = zip_ref.namelist()
                
                # Extract with progress bar
                with tqdm(total=len(files), desc="Extracting") as pbar:
                    for file in files:
                        zip_ref.extract(file, extract_dir)
                        pbar.update(1)
            
            print(f"Extracted to: {extract_dir}")

    except zipfile.BadZipFile as e:
        print(f"Error: Corrupted zip file - {e}")
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise

def main():
    """Main function to download and extract WavCaps dataset."""
    try:
        # Download dataset
        dataset_dir = download_wavcaps()
        
        # Extract all zip files
        extract_zip_files(dataset_dir)
        
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

