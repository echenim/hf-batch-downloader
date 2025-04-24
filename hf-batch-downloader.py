#!/usr/bin/env python3

import os
import time
import json
import logging
import argparse
import hashlib
from huggingface_hub import snapshot_download, HfHubHTTPError

# --------------------------
# Logging Setup
# --------------------------
def setup_logging(log_file_path: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("\ud83d\udd39 %(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# --------------------------
# Create Model Directory
# --------------------------
def create_model_dir(base: str, org: str, model: str, size: str) -> str:
    path = os.path.join(base, org, model, size)
    os.makedirs(path, exist_ok=True)
    return path

# --------------------------
# Manifest Writer
# --------------------------
def write_manifest(local_dir: str, manifest_file: str = "manifest.txt"):
    manifest_path = os.path.join(local_dir, manifest_file)
    with open(manifest_path, "w") as f:
        for root, _, files in os.walk(local_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, local_dir)
                f.write(f"{rel_path}\n")
    logger.info(f"\ud83d\udcdc Manifest exported: {manifest_path}")

# --------------------------
# Checksum Validator
# --------------------------
def verify_checksums(local_dir: str):
    for filename in os.listdir(local_dir):
        if filename.endswith(('.sha256', '.sha256sum', '.md5')):
            chk_path = os.path.join(local_dir, filename)
            logger.info(f"\ud83d\udd10 Validating: {chk_path}")
            with open(chk_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    expected, file_ref = parts[0], parts[-1].lstrip('*')
                    file_path = os.path.join(local_dir, file_ref)
                    if not os.path.exists(file_path):
                        logger.warning(f"\u26a0\ufe0f Missing: {file_ref}")
                        continue
                    with open(file_path, 'rb') as check_file:
                        data = check_file.read()
                        actual = hashlib.sha256(data).hexdigest() if filename.endswith("sha256") else hashlib.md5(data).hexdigest()
                    if actual == expected:
                        logger.info(f"\u2705 Verified: {file_ref}")
                    else:
                        logger.error(f"\u274c Checksum MISMATCH: {file_ref}")

# --------------------------
# Directory Size in GB
# --------------------------
def get_directory_size_gb(directory: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)

# --------------------------
# Download with Retry + Summary
# --------------------------
def download_model(repo_id: str, quant_patterns: list, local_dir: str, max_retries: int, backoff: int):
    attempts = 0
    start_time = time.time()

    while attempts < max_retries:
        try:
            logger.info(f"\ud83d\udce6 Downloading: {repo_id} → {quant_patterns} (Attempt {attempts + 1})")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=[f"*{q}*" for q in quant_patterns]
            )

            duration = time.time() - start_time
            write_manifest(local_dir)
            verify_checksums(local_dir)
            size_gb = get_directory_size_gb(local_dir)

            summary = (
                f"\n\u2705 Completed: {os.path.basename(local_dir)}\n"
                f"   \ud83d\udd52 Duration: {int(duration // 60)} min {int(duration % 60)} sec\n"
                f"   \ud83d\udccb Size: {size_gb:.2f} GB"
            )
            logger.info(summary)
            print(summary)
            return
        except (HfHubHTTPError, Exception) as e:
            logger.error(f"\u274c Error: {e}")

        attempts += 1
        if attempts < max_retries:
            wait = backoff * (2 ** (attempts - 1))
            logger.info(f"\u23f3 Retrying in {wait}s...")
            time.sleep(wait)
        else:
            logger.critical("\u274c Max retries reached.")

# --------------------------
# Main Entrypoint
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch download LLMs from Hugging Face with logging, manifest, retry, and checksum validation.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--base-dir", default="models", help="Base directory for downloads")
    parser.add_argument("--log", default="logs/batch_download.log", help="Log file path")
    parser.add_argument("--retries", type=int, default=3, help="Max retries")
    parser.add_argument("--backoff", type=int, default=5, help="Initial backoff in seconds")
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.log)

    with open(args.config, 'r') as f:
        models = json.load(f)

    for model in models:
        org = model["org"]
        name = model["model"]
        size = model["size"]
        repo = model["repo_id"]
        quant_list = model["quant"]

        local_dir = create_model_dir(args.base_dir, org, name, size)
        download_model(repo, quant_list, local_dir, args.retries, args.backoff)

if __name__ == "__main__":
    main()
