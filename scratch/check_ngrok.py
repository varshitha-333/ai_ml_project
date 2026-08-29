"""
Check ngrok connection to Colab GPU endpoint.
"""
import urllib.request
import ssl
import json
import time

COLAB_URL = "https://salvaging-ardently-late.ngrok-free.dev/v1/chat/completions"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request(
    COLAB_URL,
    data=json.dumps({"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8"),
    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0", "ngrok-skip-browser-warning": "true"},
    method="POST"
)

try:
    print(f"Connecting to {COLAB_URL}...")
    t0 = time.time()
    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
        t_ms = (time.time() - t0) * 1000
        print(f"SUCCESS! Connected in {round(t_ms)}ms. Status: {resp.status}")
        body = resp.read().decode("utf-8")
        print(f"Response: {body[:200]}")
except Exception as e:
    print(f"FAILED to connect to ngrok Colab URL: {e}")
