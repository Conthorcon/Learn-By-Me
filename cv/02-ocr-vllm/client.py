import base64
import requests

def perform_ocr(image_path, prompt=None):
    url = "https://20c0-34-169-136-53.ngrok-free.app/ocr"

    try:
        with open(image_path, "rb") as f:
            files = {
                "image": f
            }

            data = {}
            if prompt:
                data["prompt"] = prompt

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=60
            )

        print(f"Response time = {response.elapsed.total_seconds():.2f}s")

        response.raise_for_status()

        return response.json().get("response_message")

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None

# ===== TEST =====
result = perform_ocr("IMG_5334.JPG")
print(result)