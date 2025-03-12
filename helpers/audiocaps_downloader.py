import os
from pathlib import Path

import pandas as pd

from audiocaps_download import Downloader


dataset_split = ["train", "val", "test"]
dataset_spilt_file_rows = [1000, 200, 200]
dataset_url = \
    "https://raw.githubusercontent.com/cdjkim/audiocaps/refs/heads/master/dataset/{}.csv"
save_path = Path(__file__).parent.parent / "data" / "audiocaps"

# Ensure output directory exists
os.makedirs(save_path, exist_ok=True)

n_cores = os.cpu_count()
print(f"Using {n_cores} cores for downloading.")
# Initialize downloader using the all the cpu cores
downloader = Downloader(root_path=str(save_path), n_jobs=n_cores)  # Multi-threaded download
downloader.format = "wav"
downloader.quality = 5


# Download train, val, and test datasets
for i, split in enumerate(dataset_split):
    print(f"Downloading dataset split {split} ...")

    # Download the dataset CSV file
    df = pd.read_csv(dataset_url.format(split))

    # take the first number of rows based on the split
    num_rows = dataset_spilt_file_rows[i]
    df = df.head(num_rows)

    for index, row in df.iterrows():
        audiocap_id = row["audiocap_id"]
        youtube_id = row["youtube_id"]
        start_time = float(row["start_time"])
        end_time = start_time + 10  # Assuming each clip is 10 seconds long

        print(f"[{index+1}/{num_rows}] Downloading {audiocap_id} from {youtube_id} at {start_time}s...")

        try:
            # Download the audio file
            downloader.download_file(
                root_path=str(save_path / split),
                ytid=youtube_id,
                start_seconds=start_time,
                end_seconds=end_time,
                audiocaps_id=str(audiocap_id)
            )
        except Exception as e:
            print(f"Error downloading {audiocap_id}: {e}")

    print(f"Downloaded dataset split {split} to {save_path}")
