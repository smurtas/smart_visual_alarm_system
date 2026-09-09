from pathlib import Path

from flask import Flask, jsonify, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "runtime" / "images"
LATEST_IMAGE = IMAGE_DIR / "latest.jpg"

MAX_IMAGE_SIZE = 2 * 1024 * 1024

app = Flask(__name__)


@app.post("/upload-alert")
def upload_alert():
    if request.content_type != "image/jpeg":
        return jsonify(
            status="error",
            message="Content-Type must be image/jpeg"
        ), 415

    image_data = request.get_data()

    if not image_data:
        return jsonify(
            status="error",
            message="Empty request body"
        ), 400

    if len(image_data) > MAX_IMAGE_SIZE:
        return jsonify(
            status="error",
            message="Image too large"
        ), 413

    if not image_data.startswith(b"\xff\xd8") or not image_data.endswith(b"\xff\xd9"):
        return jsonify(
            status="error",
            message="Invalid JPEG data"
        ), 400

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_IMAGE.write_bytes(image_data)

    print(f"Image received: {len(image_data)} bytes")

    return jsonify(
        status="ok",
        bytes=len(image_data)
    ), 201


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        latest_image_exists=LATEST_IMAGE.exists()
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )