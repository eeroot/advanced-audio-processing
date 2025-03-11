import os
import pandas as pd
import torchaudio
from audiocaps_download import Downloader


"""
Lazily download audiocaps dataset per split
Install the following 
pip install audiocaps-download
pip install yt-dlp
"""
# Define paths
csv_path = "test.csv"  # CSV file containing metadata
save_path = "audiocaps/test/"  # Directory to save test files

# Load CSV and take the first 2000 rows
df = pd.read_csv(csv_path).head(2000)

# Ensure output directory exists
os.makedirs(save_path, exist_ok=True)

# Initialize downloader
downloader = Downloader(root_path=save_path, n_jobs=1)  # Single-threaded download
downloader.format = "wav" 
downloader.quality = 5  

# Iterate over the first 2000 rows in the CSV and download audio files
for index, row in df.iterrows():
    audiocap_id = row["audiocap_id"]
    youtube_id = row["youtube_id"]
    start_time = float(row["start_time"])
    end_time = start_time + 10  # Assuming each clip is 10 seconds long

    print(f"[{index+1}/2000] Downloading {audiocap_id} from {youtube_id} at {start_time}s...")

    try:
        # Download the audio file
        downloader.download_file(
            root_path=save_path,
            ytid=youtube_id,
            start_seconds=start_time,
            end_seconds=end_time,
            audiocaps_id=str(audiocap_id)
        )

        print(f"Saved: {save_path}{audiocap_id}.wav")

    except Exception as e:
        print(f"Error downloading {audiocap_id}: {e}")

print("Test dataset download complete!")
