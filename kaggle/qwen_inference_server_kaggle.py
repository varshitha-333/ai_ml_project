"""
Qwen2.5 GPU Inference Server for Kaggle Notebooks (P100 / T4 x 2 GPU).

Provides an OpenAI-compatible GPU inference API (/v1/chat/completions and /health)
running inside Kaggle Notebooks with public Ngrok tunneling.

Instructions:
1. Create a new Kaggle Notebook (Set Accelerator -> GPU P100 or T4 x 2).
2. Enable Internet under Notebook Settings (Settings -> Internet -> ON).
3. Copy and run this script in a code cell.
"""

import os
import sys
import threading
import time
import json
import torch
import requests

# 1. Install required dependencies on Kaggle
os.system("pip install -q fastapi uvicorn pyngrok bitsandbytes transformers accelerate")

import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pyngrok import ngrok

# 2. Configure Kaggle Working Directory Cache
CACHE_DIR = "/kaggle/working/huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
print(f"✅ Kaggle Model Cache Directory: {CACHE_DIR}")

# 2. Check GPU Architecture & Configure Model Loading (Prevents P100 sm_60 crash)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
is_p100 = "P100" in gpu_name

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
print(f"🚀 Loading {MODEL_ID} onto Kaggle GPU ({gpu_name})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)

if is_p100:
    print(f"Detected P100 GPU ({gpu_name}). Using torch.float32 precision to guarantee CUDA sm_60 execution...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
else:
    print(f"Detected T4 GPU ({gpu_name}). Using 4-bit bitsandbytes quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )

print("✅ SUCCESS: Qwen2.5-7B Model Loaded onto Kaggle GPU!")

# 3. FastAPI Application Setup
app = FastAPI(title="Kaggle Qwen OpenAI GPU Server")

@app.get("/health")
def health_check():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "status": "ok",
        "model_loaded": True,
        "model": MODEL_ID,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": gpu_name,
        "environment": "Kaggle GPU"
    }

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        print("\n🔥 [KAGGLE GPU] INCOMING INFERENCE REQUEST RECEIVED! 🔥")
        body = await request.json()
        messages = body.get("messages", [])
        sys_p = next((m["content"] for m in messages if m.get("role") == "system"), "")
        usr_p = next((m["content"] for m in messages if m.get("role") == "user"), "")

        prompt = f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{usr_p}<|im_end|>\n<|im_start|>assistant\n"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
            )

        text = tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"✅ [KAGGLE GPU GENERATION SUCCESS] Output ({len(text)} chars):\n{text[:150]}...\n")
        return {"choices": [{"message": {"role": "assistant", "content": text.strip()}}]}
    except Exception as e:
        print(f"❌ [KAGGLE GPU ERROR]: {e}")
        return {"choices": [{"message": {"role": "assistant", "content": "[]"}}]}

# 5. Start Uvicorn Server on 0.0.0.0 (Supports both IPv4 and IPv6)
def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

server_thread = threading.Thread(target=run_uvicorn, daemon=True)
server_thread.start()

# 6. Verify local server online
print("Waiting for Kaggle local server to come online...")
for _ in range(30):
    try:
        r = requests.get("http://127.0.0.1:8000/v1/models", timeout=1)
        if r.status_code == 200:
            print("Local server verified online!")
            break
    except Exception:
        time.sleep(1)

# 7. Ngrok Public Tunneling (Binds explicitly to 127.0.0.1:8000)
os.system("pkill -f ngrok || true")
time.sleep(1)

try:
    ngrok.kill()
    for t in ngrok.get_tunnels():
        ngrok.disconnect(t.public_url)
except Exception:
    pass

NGROK_TOKEN = os.environ.get("NGROK_AUTHTOKEN", "YOUR_NGROK_AUTHTOKEN_HERE")
if NGROK_TOKEN != "YOUR_NGROK_AUTHTOKEN_HERE":
    ngrok.set_auth_token(NGROK_TOKEN)

print("\n=======================================================")
try:
    tunnel = ngrok.connect(addr="http://127.0.0.1:8000", proto="http", bind_tls=True)
except Exception as err:
    print(f"Retrying clean Ngrok connection... ({err})")
    os.system("pkill -f ngrok || true")
    ngrok.kill()
    time.sleep(2)
    tunnel = ngrok.connect(addr="http://127.0.0.1:8000", proto="http", bind_tls=True)

print(f"  INFERENCE_URL = {tunnel.public_url}")
print("=======================================================\n")
