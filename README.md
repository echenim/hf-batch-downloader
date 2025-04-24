# 🤖 HF Batch Model Downloader

A production-grade Python script for downloading multiple Hugging Face models in batch — complete with retry logic, manifest generation, checksum validation, logging, and model size/duration summaries. Ideal for managing GGUF-format LLMs (e.g., LLaMA 4) across organizations or formats.

---

## 📦 Features

- ✅ Batch downloads across multiple repositories
- 🔁 Retry logic with exponential backoff
- 📜 Log to both console and file
- 🧾 Manifest export for all downloaded files
- 🔐 Checksum validation if `*.sha256`, `*.md5` are provided
- ⏱ Per-model download time and total size summary
- 🧩 Supports multiple quantization formats

---

## 🛠️ Installation

```bash
pip install huggingface_hub
```

---

## 🧰 Usage

### 1. Prepare your config

Create a JSON file (`models_config.json`) like so:

```json
[
  {
    "org": "meta-llama-unsloth-community",
    "model": "maverick",
    "size": "17B",
    "repo_id": "unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF",
    "quant": ["Q4_K_M", "Q5_K_M"]
  },
  {
    "org": "meta-llama-bartowski",
    "model": "scout",
    "size": "34B",
    "repo_id": "bartowski/meta-llama_Llama-4-Scout-34B-128E-Instruct-GGUF",
    "quant": ["Q4_K_M"]
  }
]
```

### 2. Run the script

```bash
python3 hf_batch_model_downloader.py \
  --config models_config.json \
  --base-dir models \
  --log logs/batch_download.log \
  --retries 3 \
  --backoff 5
```

---

## 📂 Folder Structure

```
models/
├── meta-llama-unsloth-community/
│   └── maverick/
│       └── 17B/
│           ├── *.gguf
│           ├── manifest.txt
│           └── *.sha256 (if present)
└── meta-llama-bartowski/
    └── scout/
        └── 34B/
            ├── *.gguf
            └── manifest.txt
```

---

## 🧪 Output Example

```bash
✅ Completed: 17B
   🕒 Duration: 2 min 14 sec
   💾 Size: 12.38 GB
```

---

## 🔐 Checksum Support

If any of the downloaded files include:

- `.sha256`, `.sha256sum`, or `.md5`

The script will automatically verify the checksums and log mismatches.

---

## 💡 Use Cases

- Setting up a local GGUF model hub
- Automating nightly sync of instruction-tuned models
- Model auditing with logs + manifests
- DevOps pipelines for LLM deploys

---

## 🧠 Requirements

- Python 3.7+
- `huggingface_hub`

Install:

```bash
pip install huggingface_hub
```

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).
You are free to use, modify, and distribute it, provided that modifications are also shared under the same license.

---

## 🙋‍♂️ Author

Created with ❤️ by [William Echenim](https://www.linkedin.com/in/echenim) – AI Engineer @ StrataLinks Inc.

---

## 📬 Want More?

Open an issue or feature request. Contributions welcome!
