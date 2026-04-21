import cv2
import numpy as np


def smooth_signal(signal, kernel_size=21):
    if kernel_size < 3:
        return signal.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(signal, kernel, mode="same")


def robust_std(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-6


def detect_strip_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No bright strip detected")

    h_img, w_img = gray.shape
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = h / max(w, 1)

        cx = x + w / 2
        center_distance = abs(cx - (w_img / 2))

        if area > 5000 and aspect > 1.2:
            candidates.append((cnt, area, center_distance, x, y, w, h))

    if not candidates:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
    else:
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
    h, w = strip_roi.shape[:2]

    x1 = int(0.25 * w)
    x2 = int(0.75 * w)

    y1 = int(0.35 * h)
    y2 = int(0.98 * h)

    active = strip_roi[y1:y2, x1:x2]
    return active, (x1, y1, x2, y2)


def compute_darkness_profile(active_roi):
    gray = cv2.cvtColor(active_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    row_mean = np.mean(gray, axis=1)
    row_mean_smooth = smooth_signal(row_mean, kernel_size=31)
    baseline = smooth_signal(row_mean_smooth, kernel_size=101)

    darkness = baseline - row_mean_smooth
    darkness = np.maximum(darkness, 0)

    noise_floor = np.median(darkness)
    noise_std = robust_std(darkness)

    darkness_score = (darkness - noise_floor) / noise_std
    darkness_score = np.maximum(darkness_score, 0)

    return {
        "row_mean": row_mean,
        "row_mean_smooth": row_mean_smooth,
        "baseline": baseline,
        "darkness": darkness,
        "noise_floor": float(noise_floor),
        "noise_std": float(noise_std),
        "darkness_score": darkness_score,
    }


def find_band_clusters(score_profile, threshold=2.5, min_gap=10, min_len=5):
    indices = np.where(score_profile > threshold)[0]

    if len(indices) == 0:
        return []

    splits = np.where(np.diff(indices) > min_gap)[0] + 1
    clusters = np.split(indices, splits)
    clusters = [c for c in clusters if len(c) >= min_len]

    return clusters


def summarize_bands(clusters, darkness, score_profile):
    bands = []

    for c in clusters:
        center_y = int(np.mean(c))
        width = int(len(c))

        peak_score = float(np.max(score_profile[c]))
        mean_score = float(np.mean(score_profile[c]))
        peak_darkness = float(np.max(darkness[c]))
        mean_darkness = float(np.mean(darkness[c]))
        area_score = float(np.sum(score_profile[c]))

        bands.append({
            "center_y": center_y,
            "width": width,
            "peak_score": round(peak_score, 3),
            "mean_score": round(mean_score, 3),
            "peak_darkness": round(peak_darkness, 3),
            "mean_darkness": round(mean_darkness, 3),
            "area_score": round(area_score, 3),
        })

    bands.sort(
        key=lambda b: (
            b["peak_score"] * 0.45 +
            b["mean_score"] * 0.20 +
            b["area_score"] * 0.25 +
            b["mean_darkness"] * 0.10
        ),
        reverse=True
    )

    return bands


def classify_change(bands):
    if not bands:
        return {
            "valid": True,
            "change_detected": False,
            "change_score": 0.0,
            "diagnosis": "No significant visible change detected on the strip"
        }

    strongest = bands[0]

    peak_score = strongest["peak_score"]
    mean_score = strongest["mean_score"]
    area_score = strongest["area_score"]
    width = strongest["width"]

    combined = (
        7.0 * peak_score +
        3.5 * mean_score +
        0.6 * area_score +
        0.25 * width
    )
    change_score = float(min(combined, 100.0))

    if change_score >= 60:
        diagnosis = "Strong visible change detected on the strip"
    elif change_score >= 32:
        diagnosis = "Moderate visible change detected on the strip"
    elif change_score >= 15:
        diagnosis = "Faint visible change detected on the strip"
    else:
        diagnosis = "No significant visible change detected on the strip"

    return {
        "valid": True,
        "change_detected": change_score >= 15,
        "change_score": round(change_score, 2),
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

    profile = compute_darkness_profile(active_roi)

    clusters = find_band_clusters(
        profile["darkness_score"],
        threshold=2.5,
        min_gap=10,
        min_len=5
    )

    bands = summarize_bands(
        clusters,
        profile["darkness"],
        profile["darkness_score"]
    )

    result = classify_change(bands)

    primary_band_y = bands[0]["center_y"] if bands else None
    primary_band_peak_score = bands[0]["peak_score"] if bands else 0.0
    primary_band_mean_darkness = bands[0]["mean_darkness"] if bands else 0.0
    primary_band_peak_darkness = bands[0]["peak_darkness"] if bands else 0.0

    # Realistic user-facing intensity: scaled from raw darkness, not normalized to 1.0
    # Clamp to a practical 0-100 range for UI display.
    realistic_intensity = min(primary_band_mean_darkness * 8.0, 100.0)

    result.update({
        "detected_band_count": len(bands),
        "primary_band_y": primary_band_y,
        "primary_band_peak_score": round(primary_band_peak_score, 4),
        "primary_band_mean_darkness": round(primary_band_mean_darkness, 4),
        "primary_band_peak_darkness": round(primary_band_peak_darkness, 4),
        "realistic_intensity": round(realistic_intensity, 2),
        "bands": bands[:3],
        "active_region_box": {
            "x1": int(active_box[0]),
            "y1": int(active_box[1]),
            "x2": int(active_box[2]),
            "y2": int(active_box[3]),
        },
        "noise_floor": round(profile["noise_floor"], 4),
        "noise_std": round(profile["noise_std"], 4),
    })

    return result


