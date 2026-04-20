import os
import subprocess

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def capture_image(filename="test.jpg"):
    filepath = os.path.join(RAW_DIR, filename)

    cmd = [
        "rpicam-jpeg",
        "-o", filepath,
        "--width", "2304",
        "--height", "1296",
        "-t", "1000"
    ]
    subprocess.run(cmd, check=True)
    return filepath

