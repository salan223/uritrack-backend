import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Image paths
CAPTURE_IMAGE_PATH = os.path.join(RAW_DIR, "test.jpg")
DEBUG_IMAGE_PATH = os.path.join(PROCESSED_DIR, "debug.jpg")

# Camera settings
CAMERA_WARMUP_SECONDS = 2

# Temporary fixed ROI for first testing
# Format: gray[y1:y2, x1:x2]
ROI_Y1 = 200
ROI_Y2 = 300
ROI_X1 = 400
ROI_X2 = 600

# Simple classification thresholds
STRONG_THRESHOLD = 120
MODERATE_THRESHOLD = 180