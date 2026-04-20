import cv2
import numpy as np


def smooth_signal(signal, kernel_size=21):
    if kernel_size < 3:
        return signal.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(signal, kernel, mode="same")


def detect_strip_roi(image):
    """
    Detect the bright strip against the darker background.
    Returns a cropped ROI containing the strip.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bright strip threshold
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No bright strip detected")

    h_img, w_img = gray.shape
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = h / max(w, 1)

        # Prefer tall bright objects near the center
        cx = x + w / 2
        center_distance = abs(cx - (w_img / 2))

        if area > 5000 and aspect > 1.2:
            candidates.append((cnt, area, center_distance, x, y, w, h))

    if not candidates:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
    else:
        # Bigger area and closer to center is better
        candidates.sort(key=lambda item: (-item[1], item[2]))
        _, _, _, x, y, w, h = candidates[0]

    pad_x = int(0.08 * w)
    pad_y = int(0.03 * h)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    roi = image[y1:y2, x1:x2]
    return roi


def extract_active_strip_region(strip_roi):
    """
    Focus analysis on the lower central portion of the strip where visible changes/bands occur.
    """
    h, w = strip_roi.shape[:2]

    # Ignore outer edges; use central width
    x1 = int(0.25 * w)
    x2 = int(0.75 * w)

    # Ignore top handle/cassette area; focus more on lower portion
    y1 = int(0.35 * h)
    y2 = int(0.98 * h)

    active = strip_roi[y1:y2, x1:x2]
    return active, (x1, y1, x2, y2)


def compute_darkness_profile(active_roi):
    """
    Compute a row-wise darkness/change profile on the bright strip.
    A darker band on a white strip creates a positive peak in this profile.
    """
    gray = cv2.cvtColor(active_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Average brightness for each row
    row_mean = np.mean(gray, axis=1)

    # Smooth raw brightness
    row_mean_smooth = smooth_signal(row_mean, kernel_size=31)

    # Local baseline: broader smooth estimate of the white strip brightness
    baseline = smooth_signal(row_mean_smooth, kernel_size=101)

    # Darkness score = how much darker this row is than local baseline
    darkness = baseline - row_mean_smooth

    # Clamp negatives
    darkness = np.maximum(darkness, 0)

    # Normalize if possible
    max_val = float(np.max(darkness))
    if max_val > 1e-6:
        darkness_norm = darkness / max_val
    else:
        darkness_norm = darkness.copy()

    return row_mean_smooth, baseline, darkness, darkness_norm


def find_band_clusters(darkness_norm, threshold=0.22, min_gap=10, min_len=4):
    """
    Find clusters of rows where darkness exceeds threshold.
    These clusters correspond to visible changes/bands on the strip.
    """
    indices = np.where(darkness_norm > threshold)[0]

    if len(indices) == 0:
        return []

    splits = np.where(np.diff(indices) > min_gap)[0] + 1
    clusters = np.split(indices, splits)
    clusters = [c for c in clusters if len(c) >= min_len]

    return clusters


def summarize_bands(clusters, darkness, darkness_norm):
    """
    Turn row clusters into band summaries.
    """
    bands = []

    for c in clusters:
        center_y = int(np.mean(c))
        width = int(len(c))
        strength_raw = float(np.mean(darkness[c]))
        strength_norm = float(np.mean(darkness_norm[c]))
        peak_norm = float(np.max(darkness_norm[c]))

        bands.append({
            "center_y": center_y,
            "width": width,
            "strength_raw": round(strength_raw, 3),
            "strength_norm": round(strength_norm, 4),
            "peak_norm": round(peak_norm, 4),
        })

    bands.sort(key=lambda b: b["peak_norm"], reverse=True)
    return bands


def classify_change(bands):
    """
    Convert detected band strength into a simple interpretation.
    This does not assume a specific assay layout; it just measures visible change.
    """
    if not bands:
        return {
            "valid": True,
            "change_detected": False,
            "change_score": 0.0,
            "diagnosis": "No significant visible change detected on the strip"
        }

    strongest = bands[0]
    score = float(strongest["peak_norm"]) * 100.0

    if score >= 70:
        diagnosis = "Strong visible change detected on the strip"
    elif score >= 40:
        diagnosis = "Moderate visible change detected on the strip"
    elif score >= 18:
        diagnosis = "Faint visible change detected on the strip"
    else:
        diagnosis = "No significant visible change detected on the strip"

    return {
        "valid": True,
        "change_detected": score >= 18,
        "change_score": round(score, 2),
        "diagnosis": diagnosis
    }


def analyze_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    strip_roi = detect_strip_roi(image)
    active_roi, active_box = extract_active_strip_region(strip_roi)

    if active_roi.size == 0:
        return {
            "valid": False,
            "diagnosis": "Active strip region could not be extracted"
        }

    row_mean, baseline, darkness, darkness_norm = compute_darkness_profile(active_roi)
    clusters = find_band_clusters(
        darkness_norm,
        threshold=0.22,
        min_gap=10,
        min_len=4
    )
    bands = summarize_bands(clusters, darkness, darkness_norm)

    result = classify_change(bands)

    primary_band_y = bands[0]["center_y"] if bands else None
    primary_band_strength = bands[0]["peak_norm"] if bands else 0.0

    result.update({
        "detected_band_count": len(bands),
        "primary_band_y": primary_band_y,
        "primary_band_strength": round(primary_band_strength, 4),
        "bands": bands[:3],  # top 3 only
        "active_region_box": {
            "x1": int(active_box[0]),
            "y1": int(active_box[1]),
            "x2": int(active_box[2]),
            "y2": int(active_box[3]),
        }
    })

    return result


