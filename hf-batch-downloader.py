import os
import time
import json
import logging
import argparse
import shutil
import hashlib
from huggingface_hub import snapshot_download

# --------------------------
# Logging Setup
# --------------------------
def setup_logging(log_file_path: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("🔹 %(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

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
# Disk Space Checker
# --------------------------
def check_disk_space(path: str, required_gb: float = 50.0):
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(f"❌ Not enough disk space. Required: {required_gb:.2f} GB, Available: {free_gb:.2f} GB")
    logger.info(f"✅ Sufficient disk space: {free_gb:.2f} GB free")

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
    logger.info(f"📝 Manifest exported: {manifest_path}")

# --------------------------
# Checksum Validator
# --------------------------
def verify_checksums(local_dir: str):
    for filename in os.listdir(local_dir):
        if filename.endswith(('.sha256', '.sha256sum', '.md5')):
            chk_path = os.path.join(local_dir, filename)
            logger.info(f"🔐 Validating: {chk_path}")
            with open(chk_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    expected, file_ref = parts[0], parts[-1].lstrip('*')
                    file_path = os.path.join(local_dir, file_ref)
                    if not os.path.exists(file_path):
                        logger.warning(f"⚠️ Missing: {file_ref}")
                        continue
                    with open(file_path, 'rb') as check_file:
                        data = check_file.read()
                        actual = hashlib.sha256(data).hexdigest() if filename.endswith("sha256") else hashlib.md5(data).hexdigest()
                    if actual == expected:
                        logger.info(f"✅ Verified: {file_ref}")
                    else:
                        logger.error(f"❌ Checksum MISMATCH: {file_ref}")

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
def download_model(repo_id: str, quant_patterns: list, local_dir: str, max_retries: int, backoff: int, hf_home: str, min_disk: float):
    if not os.path.exists(hf_home):
        try:
            os.makedirs(hf_home, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"❌ Cannot create Hugging Face cache directory at '{hf_home}'. Try running with elevated permissions or use a writable path.")
    check_disk_space(hf_home, required_gb=min_disk)
    os.environ["HF_HOME"] = hf_home

    attempts = 0
    start_time = time.time()

    while attempts < max_retries:
        try:
            logger.info(f"📦 Downloading: {repo_id} → {quant_patterns} (Attempt {attempts + 1})")
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
                f"\n✅ Completed: {os.path.basename(local_dir)}\n"
                f"   🕒 Duration: {int(duration // 60)} min {int(duration % 60)} sec\n"
                f"   💾 Size: {size_gb:.2f} GB"
            )
            logger.info(summary)
            print(summary)
            return
        except Exception as e:
            logger.error(f"❌ Error: {e}")

        attempts += 1
        if attempts < max_retries:
            wait = backoff * (2 ** (attempts - 1))
            logger.info(f"⏳ Retrying in {wait}s...")
            time.sleep(wait)
        else:
            logger.critical("❌ Max retries reached.")

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
    parser.add_argument("--hf-home", default=os.path.expanduser("~/hf_cache"), help="Custom Hugging Face cache directory (default: ~/hf_cache)")
    parser.add_argument("--min-disk", type=float, default=60.0, help="Minimum free disk space in GB required before download")
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
        download_model(repo, quant_list, local_dir, args.retries, args.backoff, args.hf_home, args.min_disk)

if __name__ == "__main__":
    main()
