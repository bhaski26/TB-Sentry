from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import os
import sys

MODEL_URL = (
    "https://github.com/bhaski26/TB-Sentry/releases/download/"
    "v1.0.0/densenet121_baseline.pt"
)

MODEL_PATH = Path("models/checkpoints/densenet121_baseline.pt")

EXPECTED_SHA256 = (
    "6a8f947df1fe3723a801b1c88d4bc8b7cc91ca03e1915ffe4cb30a450031fa22"
)

CHUNK_SIZE = 1024 * 1024


def calculate_sha256(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def download_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print("Existing model checkpoint found.")

        existing_hash = calculate_sha256(MODEL_PATH)

        if existing_hash == EXPECTED_SHA256:
            print("SHA-256 verification successful.")
            print(f"Model ready: {MODEL_PATH}")
            return

        print("Existing checkpoint failed verification.")
        print("Downloading a verified copy...")

    temporary_path = MODEL_PATH.with_suffix(".pt.download")

    try:
        print("Downloading TB-Sentry model checkpoint...")
        print(f"Source: {MODEL_URL}")

        request = Request(
            MODEL_URL,
            headers={"User-Agent": "TB-Sentry-Model-Downloader"}
        )

        downloaded = 0

        with urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))

            with temporary_path.open("wb") as file:
                while True:
                    chunk = response.read(CHUNK_SIZE)

                    if not chunk:
                        break

                    file.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        percentage = downloaded * 100 / total
                        print(
                            f"\rProgress: {percentage:.1f}%",
                            end="",
                            flush=True
                        )

        print()

        actual_hash = calculate_sha256(temporary_path)

        if actual_hash != EXPECTED_SHA256:
            raise RuntimeError(
                "SHA-256 verification failed. "
                "The downloaded file may be corrupted."
            )

        os.replace(temporary_path, MODEL_PATH)

        print("SHA-256 verification successful.")
        print(f"Model downloaded successfully: {MODEL_PATH}")

    except Exception as error:
        if temporary_path.exists():
            temporary_path.unlink()

        print(f"ERROR: Model download failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()