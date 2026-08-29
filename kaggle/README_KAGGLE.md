# Kaggle GPU Inference Server Setup Guide

This guide explains how to run the **Qwen2.5-7B-Instruct** open-weight GPU inference server inside a **Kaggle GPU Notebook** (P100 / T4 x 2 GPUs).

---

## 📋 Step-by-Step Kaggle Setup

### 1. Create a New Kaggle Notebook
1. Log into your [Kaggle Account](https://www.kaggle.com/).
2. Click **+ Create -> New Notebook**.

### 2. Enable GPU & Internet Access (CRITICAL)
1. In the right panel under **Notebook Settings**:
   - **Accelerator**: Select **GPU P100** or **GPU T4 x 2**.
   - **Internet**: Toggle to **ON** (Required to download HuggingFace weights).

### 3. Run the Inference Server
1. Copy the contents of `kaggle/qwen_inference_server_kaggle.py` into a code cell in your Kaggle notebook.
2. Click **Run Cell** (Shift + Enter).

### 4. Copy the Generated `INFERENCE_URL`
The cell will output:
```text
=======================================================
  INFERENCE_URL = https://xxxx-xx-xx-xx.ngrok-free.dev
  Copy this INFERENCE_URL into your Docker command!
=======================================================
```
Copy this URL into your environment variable:
```powershell
$env:INFERENCE_URL = "https://xxxx-xx-xx-xx.ngrok-free.dev"
```

---

## 🚀 Persistent Caching on Kaggle

The script automatically configures HuggingFace model cache under `/kaggle/working/huggingface_cache`. As long as your Kaggle notebook session is active, Qwen model weights remain cached in memory for instant execution!
