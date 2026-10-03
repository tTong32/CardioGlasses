"""Download the PhysioNet frailty dataset."""
import subprocess
import sys
from pathlib import Path


def download_dataset():
    """Download the wearable-exercise-frailty dataset from PhysioNet."""
    data_dir = Path("data/external/frailty")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    expected_files = ["subject-info.csv", "test-availability.csv"]
    if all((data_dir / f).exists() for f in expected_files):
        print(f"Dataset already present in {data_dir}")
        print("\nContents:")
        for item in sorted(data_dir.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.name}: {size_mb:.2f} MB")
            elif item.is_dir():
                n_files = len(list(item.iterdir()))
                print(f"  {item.name}/: {n_files} files")
        return

    print(f"Downloading PhysioNet frailty dataset to {data_dir}...")

    # Try AWS S3 first (faster, no rate limits)
    print("\nAttempting AWS S3 download...")
    s3_cmd = [
        "aws", "s3", "sync",
        "--no-sign-request",
        "s3://physionet-open/wearable-exercise-frailty/1.0.0/",
        str(data_dir)
    ]

    try:
        result = subprocess.run(s3_cmd, check=True, capture_output=True, text=True)
        print("AWS S3 download succeeded!")
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"AWS S3 download failed ({e}), falling back to wget...")

        # Fallback to wget
        wget_cmd = [
            "wget",
            "-r",  # recursive
            "-N",  # timestamping
            "-c",  # continue
            "-np",  # no parent
            "-nH",  # no host directories
            "--cut-dirs=3",
            "-P", str(data_dir),
            "https://physionet.org/files/wearable-exercise-frailty/1.0.0/"
        ]

        try:
            subprocess.run(wget_cmd, check=True)
            print("wget download succeeded!")
        except (subprocess.CalledProcessError, FileNotFoundError) as e2:
            print(f"ERROR: Both download methods failed.", file=sys.stderr)
            print(f"  AWS S3: {e}", file=sys.stderr)
            print(f"  wget: {e2}", file=sys.stderr)
            sys.exit(1)

    # Report what was downloaded
    print(f"\nDownload complete! Contents of {data_dir}:")
    for item in sorted(data_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name}: {size_mb:.2f} MB")
        elif item.is_dir():
            n_files = len(list(item.iterdir()))
            print(f"  {item.name}/: {n_files} files")


if __name__ == "__main__":
    download_dataset()
