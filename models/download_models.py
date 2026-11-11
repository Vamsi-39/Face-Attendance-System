import os
import urllib.request
import sys

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "description": "YuNet CNN Face Detector (OpenCV Zoo)",
        "min_size_kb": 200
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "description": "SFace CNN Face Recognizer (OpenCV Zoo)",
        "min_size_kb": 30000
    }
}

def get_models_dir() -> str:
    """Returns absolute path to models directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return current_dir

def download_models(force: bool = False) -> dict:
    models_dir = get_models_dir()
    os.makedirs(models_dir, exist_ok=True)
    paths = {}

    for filename, info in MODELS.items():
        dest_path = os.path.join(models_dir, filename)
        paths[filename] = dest_path

        if os.path.exists(dest_path) and not force:
            size_kb = os.path.getsize(dest_path) / 1024
            if size_kb >= info["min_size_kb"]:
                print(f"[OK] {filename} already exists ({size_kb:.1f} KB).")
                continue
            else:
                print(f"[!] {filename} file seems incomplete ({size_kb:.1f} KB). Re-downloading...")

        print(f"[DOWNLOAD] Downloading {info['description']}...")
        try:
            req = urllib.request.Request(
                info["url"],
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 64
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        sys.stdout.write(f"\r    Progress: {percent:5.1f}% ({downloaded // 1024} KB / {total_size // 1024} KB)")
                        sys.stdout.flush()
            print(f"\n[OK] Successfully downloaded {filename}")
        except Exception as e:
            print(f"\n[ERROR] Failed to download {filename}: {e}")
            if os.path.exists(dest_path):
                try: os.remove(dest_path)
                except: pass
            raise e

    return paths

if __name__ == "__main__":
    download_models()
