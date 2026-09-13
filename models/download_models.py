"""
Downloads the model files CineMind AI's mood-detection feature needs but
that were never included in the project folder:

    models/deploy.prototxt
    models/res10_300x300_ssd_iter_140000.caffemodel
    models/emotion-ferplus-8.onnx

Run this once from anywhere (it writes into its own folder):

    python models/download_models.py

No extra packages required - just the Python standard library.
"""
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
    "emotion-ferplus-8.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    ),
}

# Fallback mirror in case the ONNX model-zoo path above ever moves.
ONNX_FALLBACK_URL = (
    "https://github.com/onnx/models/raw/main/vision/body_analysis/"
    "emotion_ferplus/model/emotion-ferplus-8.onnx"
)


def download(name, url):
    dest = MODEL_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {name} already exists ({dest.stat().st_size:,} bytes)")
        return
    print(f"[get ] {name}  <-  {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[done] {name}  ({dest.stat().st_size:,} bytes)")
    except Exception as exc:
        if name == "emotion-ferplus-8.onnx":
            print(f"[retry] primary URL failed ({exc}); trying fallback mirror...")
            try:
                urllib.request.urlretrieve(ONNX_FALLBACK_URL, dest)
                print(f"[done] {name}  ({dest.stat().st_size:,} bytes)")
                return
            except Exception as exc2:
                print(f"[fail] {name}: {exc2}")
                return
        print(f"[fail] {name}: {exc}")


if __name__ == "__main__":
    print(f"Downloading model files into: {MODEL_DIR}\n")
    for filename, url in FILES.items():
        download(filename, url)
    print("\nDone. Re-run the app - the Mood Recommendations camera "
          "detector should now work:\n    streamlit run app.py")
