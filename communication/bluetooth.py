import json
import base64
import bluetooth
import cv2


def encode_image_to_base64(image_path: str, max_width: int = 640, jpeg_quality: int = 60) -> str:
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))

    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not success:
        raise RuntimeError("Failed to encode image")

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def send_result_over_bluetooth(payload: dict):
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", 1))
    server_sock.listen(1)

    print("Bluetooth server waiting on RFCOMM channel 1...")
    print("Open Serial Bluetooth Terminal on your phone and connect.")

    client_sock = None
    try:
        client_sock, client_info = server_sock.accept()
        print(f"Bluetooth client connected: {client_info}")

        image_path = payload.get("image_path")
        if image_path:
            payload["image_base64"] = encode_image_to_base64(image_path)

        message = json.dumps(payload) + "\n"
        client_sock.send(message.encode("utf-8"))

        print("Bluetooth data sent successfully.")
    finally:
        if client_sock is not None:
            client_sock.close()
        server_sock.close()
