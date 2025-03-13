import os
import sys
import argparse

from pathlib import Path

import pandas as pd

from audiocaps_download import Downloader


# Set up argument parser
parser = argparse.ArgumentParser(description='Download AudioCaps dataset')
parser.add_argument('--train-size', type=int, default=1000, help='Number of training samples to download')
parser.add_argument('--val-size', type=int, default=200, help='Number of validation samples to download')
parser.add_argument('--test-size', type=int, default=200, help='Number of test samples to download')
parser.add_argument('--extension', type=str, default='wav', help='File extension to download')

args = parser.parse_args()

dataset_split = ["train", "val", "test"]
dataset_spilt_sizes = [args.train_size, args.val_size, args.test_size]
dataset_url = \
    "https://raw.githubusercontent.com/cdjkim/audiocaps/refs/heads/master/dataset/{}.csv"
save_path = Path(__file__).parent.parent / "data" / "audiocaps"

# Ensure output directory exists
os.makedirs(save_path, exist_ok=True)

n_cores = os.cpu_count()
print(f"Using {n_cores} cores for downloading.")
# Initialize downloader using the all the cpu cores
downloader = Downloader(root_path=str(save_path), n_jobs=n_cores)  # Multi-threaded download
downloader.format = args.extension
downloader.quality = 5


# Download train, val, and test datasets
for i, split in enumerate(dataset_split):
    print(f"Downloading dataset split {split} ...")

    # Download the dataset CSV file
    df = pd.read_csv(dataset_url.format(split))

    # take the first number of rows based on the split
    num_samples = dataset_spilt_sizes[i]

    # create a new empty dataframe
    out_rows = []

    for _, row in df.iterrows():
        # get length of out_df dataframe
        index = len(out_rows)
        if index == num_samples:
            break

        audiocap_id = row["audiocap_id"]
        youtube_id = row["youtube_id"]
        start_time = float(row["start_time"])
        end_time = start_time + 10  # Assuming each clip is 10 seconds long

        print(f"[{index}/{num_samples}] Downloading {audiocap_id} from {youtube_id} at {start_time}s...")

        # Check if file already exists
        output_file = save_path / split / f"{audiocap_id}.{args.extension}"
        if not output_file.exists():
            # Download the audio file
            downloader.download_file(
                root_path=str(save_path / split),
                ytid=youtube_id,
                start_seconds=start_time,
                end_seconds=end_time,
                audiocaps_id=str(audiocap_id)
            )
            if not output_file.exists():
                continue
        else:
            print(f"File {output_file} already exists, skipping...")

        # write the row to the new dataframe
        out_rows.append(row)


    print(f"Downloaded dataset split {split} to {save_path}")
    # Save the new dataframe to a CSV file
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(save_path / f"{split}.csv", index=False)
