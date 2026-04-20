from flask import Flask, jsonify, send_from_directory
from datetime import datetime
import json
import os

from camera.capture import capture_image
from processing.analyze import analyze_image

app = Flask(__name__)

RESULTS_DIR = "data/results"
RAW_DIR = "data/raw"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

LATEST_RESULT_PATH = os.path.join(RESULTS_DIR, "latest_result.json")
RESULTS_LOG_PATH = os.path.join(RESULTS_DIR, "results_log.jsonl")


def save_result(result: dict):
    with open(LATEST_RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    with open(RESULTS_LOG_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "message": "Kidney stone detector API is running"
    })


@app.route("/image/<filename>", methods=["GET"])
def get_image(filename):
    return send_from_directory(RAW_DIR, filename)


@app.route("/latest-result", methods=["GET"])
def latest_result():
    if not os.path.exists(LATEST_RESULT_PATH):
        return jsonify({
            "ok": False,
            "error": "No result available yet"
        }), 404

    with open(LATEST_RESULT_PATH, "r") as f:
        data = json.load(f)

    return jsonify(data)


@app.route("/capture-analyze", methods=["GET"])
def capture_and_analyze():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"

        image_path = capture_image(filename)
        analysis = analyze_image(image_path)

        pi_ip = "10.0.0.119"

        result = {
            "ok": True,
            "timestamp": timestamp,
            "filename": filename,
            "image_path": image_path,
            "image_url": f"http://{pi_ip}:8000/image/{filename}",
            "analysis": analysis
        }

        save_result(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


