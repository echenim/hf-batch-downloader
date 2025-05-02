#!/usr/bin/env python3
"""
Batch download LLM models from Hugging Face with logging, manifest, retry, checksum validation,
concurrency, progress reporting, automatic merging of split GGUF files (with cleanup), and optional skip flags.
"""
import os
import time
import json
import logging
import argparse
import shutil
import hashlib
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from huggingface_hub import snapshot_download

# --------------------------
# Logging Setup
# --------------------------
def setup_logging(log_file_path: str) -> logging.Logger:
    logger = logging.getLogger("hf_batch_downloader")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("🔹 %(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file handler
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

# --------------------------
# Helpers
# --------------------------
def create_model_dir(base: str, org: str, model: str, size: str) -> str:
    path = os.path.join(base, org, model, size)
    os.makedirs(path, exist_ok=True)
    return path


def check_disk_space(path: str, required_gb: float = 50.0):
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(f"❌ Not enough disk space: required {required_gb:.2f} GB, available {free_gb:.2f} GB")


def write_manifest(local_dir: str, logger: logging.Logger):
    manifest_path = os.path.join(local_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        for root, _, files in os.walk(local_dir):
            for filename in files:
                rel = os.path.relpath(os.path.join(root, filename), local_dir)
                f.write(rel + "\n")
    logger.info(f"📝 Manifest written: {manifest_path}")


def verify_checksums(local_dir: str, logger: logging.Logger):
    files = [f for f in os.listdir(local_dir) if f.endswith(('.sha256', '.sha256sum', '.md5'))]
    for chk in files:
        chk_path = os.path.join(local_dir, chk)
        logger.info(f"🔐 Verifying: {chk_path}")
        with open(chk_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                expected, ref = parts[0], parts[-1].lstrip('*')
                target = os.path.join(local_dir, ref)
                if not os.path.exists(target):
                    logger.warning(f"⚠️ Missing file for checksum: {ref}")
                    continue
                h = hashlib.sha256() if chk.endswith('sha256') else hashlib.md5()
                with open(target, 'rb') as tf:
                    for chunk in iter(lambda: tf.read(8192), b""):
                        h.update(chunk)
                actual = h.hexdigest()
                if actual == expected:
                    logger.info(f"✅ Checksum OK: {ref}")
                else:
                    logger.error(f"❌ Checksum mismatch: {ref} (expected {expected}, got {actual})")


def get_dir_size_gb(path: str) -> float:
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(dp, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


def merge_gguf_parts(local_dir: str, logger: logging.Logger):
    # Find split parts: pattern includes '-of-'
    pattern = os.path.join(local_dir, "*-of-*.gguf")
    parts = sorted(glob.glob(pattern))
    if len(parts) <= 1:
        return None
    # Derive output name: prefix before '00001-of-'
    base_name = os.path.basename(parts[0])
    prefix = base_name.split('-00001-of-')[0]
    full_name = f"{prefix}-full.gguf"
    full_path = os.path.join(local_dir, full_name)
    # Merge with progress
    logger.info(f"🔀 Merging {len(parts)} parts into {full_name}")
    with open(full_path, 'wb') as outfile:
        for part in tqdm(parts, desc="Merging parts", unit="file", leave=False):
            with open(part, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    logger.info(f"✅ Merged file created: {full_path}")
    # Cleanup with progress
    for part in tqdm(parts, desc="Deleting parts", unit="file", leave=False):
        try:
            os.remove(part)
            logger.info(f"🗑️ Deleted part: {part}")
        except Exception as e:
            logger.warning(f"⚠️ Could not delete part {part}: {e}")
    return full_path

# --------------------------
# Download Function
# --------------------------
def download_model(
    repo_id: str,
    quant_patterns: list,
    local_dir: str,
    max_retries: int,
    backoff: int,
    hf_home: str,
    min_disk: float,
    skip_manifest: bool,
    skip_verify: bool,
    logger: logging.Logger,
) -> dict:
    os.environ["HF_HOME"] = hf_home
    os.makedirs(hf_home, exist_ok=True)
    check_disk_space(hf_home, min_disk)

    start = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"📦 Downloading {repo_id} patterns={quant_patterns} (try {attempt}/{max_retries})")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=[f"*{q}*" for q in quant_patterns]
            )
            if not skip_manifest:
                write_manifest(local_dir, logger)
            if not skip_verify:
                verify_checksums(local_dir, logger)

            # Merge split GGUF parts if present (and cleanup)
            merged = merge_gguf_parts(local_dir, logger)

            size = get_dir_size_gb(local_dir)
            elapsed = time.time() - start
            result = {"repo": repo_id, "status": "ok", "duration": elapsed, "size_gb": size}
            if merged:
                result["merged_file"] = merged
            return result
        except Exception as e:
            logger.error(f"❌ Error downloading {repo_id}: {e}")
            if attempt < max_retries:
                wait = backoff * (2 ** (attempt - 1))
                logger.info(f"⏳ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return {"repo": repo_id, "status": "failed", "error": str(e)}

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch download LLMs from Hugging Face with advanced features."
    )
    parser.add_argument("--config", required=True, help="JSON config path")
    parser.add_argument("--base-dir", default="models", help="Download root directory")
    parser.add_argument("--log", default="logs/batch_download.log", help="Log file path")
    parser.add_argument("--retries", type=int, default=3, help="Max download retries")
    parser.add_argument("--backoff", type=int, default=5, help="Initial backoff seconds")
    parser.add_argument("--hf-home", default=os.path.expanduser("~/hf_cache"), help="HF cache dir")
    parser.add_argument("--min-disk", type=float, default=60.0, help="Min free disk GB")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent downloads")
    parser.add_argument("--skip-manifest", action="store_true", help="Skip manifest writing")
    parser.add_argument("--skip-verify", action="store_true", help="Skip checksum validation")
    args = parser.parse_args()

    logger = setup_logging(args.log)
    with open(args.config) as cf:
        models = json.load(cf)

    tasks = []
    for m in models:
        d = create_model_dir(args.base_dir, m["org"], m["model"], m["size"])
        tasks.append((m["repo_id"], m["quant"], d))

    results = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    download_model,
                    repo, patterns, ld,
                    args.retries, args.backoff,
                    args.hf_home, args.min_disk,
                    args.skip_manifest, args.skip_verify,
                    logger
                ): repo for repo, patterns, ld in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Models"):
                results.append(fut.result())
    else:
        for repo, patterns, ld in tqdm(tasks, desc="Models"):
            results.append(
                download_model(
                    repo, patterns, ld,
                    args.retries, args.backoff,
                    args.hf_home, args.min_disk,
                    args.skip_manifest, args.skip_verify,
                    logger
                )
            )

    logger.info("📊 Summary:")
    for r in results:
        if r.get("status") == "ok":
            msg = f"✅ {r['repo']} in {int(r['duration']//60)}m {int(r['duration']%60)}s, {r['size_gb']:.2f}GB"
            if r.get("merged_file"):
                msg += f", merged into {r['merged_file']}"
            logger.info(msg)
        else:
            logger.error(f"❌ {r['repo']} failed: {r.get('error')}" )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user, exiting.")
        exit(1)
