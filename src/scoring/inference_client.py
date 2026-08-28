"""
InferenceClient Abstraction Layer for HTTP communications with remote/local
OpenAI-compatible inference endpoints (vLLM, Ollama, TGI, or Colab Tunnels).
"""

import os
import json
import time
import socket
import urllib.request
import urllib.error
from typing import Tuple, Optional


class InferenceError(Exception):
    """Base exception for inference client failures."""
    pass


class InferenceTimeoutError(InferenceError):
    """Raised when an inference read operation times out."""
    pass


class InferenceConnectionError(InferenceError):
    """Raised when connecting to the inference endpoint fails."""
    pass


class InferenceHTTPError(InferenceError):
    """Raised when the inference endpoint returns an HTTP 4xx or 5xx error."""
    pass


class InferenceClient:
    """
    HTTP client abstraction for OpenAI-compatible chat completion endpoints.
    Decouples scoring pipeline from specific deployment infrastructure.
    """

    def __init__(
        self,
        inference_url: Optional[str] = None,
        model_name: Optional[str] = None,
        connect_timeout: Optional[int] = None,
        read_timeout: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: int = 1
    ):
        raw_url = (inference_url or os.getenv("INFERENCE_URL", "http://localhost:8000")).rstrip("/")
        for suffix in ["/v1/chat/completions", "/v1/models", "/v1"]:
            if raw_url.endswith(suffix):
                raw_url = raw_url[:-len(suffix)].rstrip("/")
        self.inference_url = raw_url
        self.model_name = model_name or os.getenv("INFERENCE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        
        self.connect_timeout = connect_timeout or int(os.getenv("INFERENCE_CONNECT_TIMEOUT", "10"))
        # Support both 'read_timeout' and backward compatible 'timeout'
        eff_read = read_timeout or timeout or int(os.getenv("INFERENCE_READ_TIMEOUT", "60"))
        self.read_timeout = eff_read
        self.max_retries = max_retries

    def health_check(self) -> Tuple[bool, str]:
        """
        Queries endpoint status. Returns (is_healthy, status_message).
        """
        url = f"{self.inference_url}/v1/models"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FacetEvaluatorClient/1.0"})
            with urllib.request.urlopen(req, timeout=self.connect_timeout) as resp:
                if resp.status == 200:
                    return True, "Inference endpoint online and healthy"
                return False, f"Endpoint returned status code {resp.status}"
        except urllib.error.URLError as err:
            return False, f"Inference endpoint unreachable at {self.inference_url}: {err.reason}"
        except Exception as err:
            return False, f"Health check failed: {str(err)}"

    def generate(self, system_prompt: str, user_prompt: str, request_id: str = "req_1") -> str:
        """
        Generates completion string from remote inference endpoint with controlled timeout error handling.
        Does NOT retry on read timeouts to prevent duplicate expensive LLM generations.
        """
        endpoint = f"{self.inference_url}/v1/chat/completions"
        start_time = time.time()
        
        print(f"[INFERENCE] request_id={request_id} endpoint={endpoint} model={self.model_name} start")

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 180
        }
        data = json.dumps(payload).encode("utf-8")

        for attempt in range(1, self.max_retries + 2):
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "FacetEvaluatorClient/1.0"
                    }
                )
                
                # Use configured read_timeout for HTTP request execution
                with urllib.request.urlopen(req, timeout=self.read_timeout) as resp:
                    latency_ms = int((time.time() - start_time) * 1000)
                    if resp.status == 200:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        choices = res_json.get("choices", [])
                        if choices and "message" in choices[0]:
                            completion_len = len(choices[0]["message"]["content"])
                            print(f"[INFERENCE] request_id={request_id} HTTP_status=200 latency_ms={latency_ms} response_chars={completion_len}")
                            return choices[0]["message"]["content"].strip()
                        raise ValueError("Invalid OpenAI API response structure: missing choices[0].message")
                    else:
                        print(f"[INFERENCE] request_id={request_id} HTTP_status={resp.status} latency_ms={latency_ms}")
                        raise InferenceHTTPError(f"HTTP status code {resp.status}")

            except (socket.timeout, TimeoutError):
                latency_ms = int((time.time() - start_time) * 1000)
                print(f"[INFERENCE] request_id={request_id} ERROR=ReadTimeout latency_ms={latency_ms}")
                # DO NOT RETRY READ TIMEOUTS: Repeating expensive LLM generation creates latency cascade
                raise InferenceTimeoutError(f"Inference read operation timed out after {self.read_timeout}s.")

            except urllib.error.HTTPError as err:
                latency_ms = int((time.time() - start_time) * 1000)
                print(f"[INFERENCE] request_id={request_id} ERROR=HTTP_{err.code} reason='{err.reason}' latency_ms={latency_ms}")
                if err.code == 404:
                    print(f"❌ [INFERENCE_CLIENT ERROR] HTTP 404 Not Found at {endpoint}.")
                    print(f"👉 TIP: Your Ngrok tunnel URL in Colab may have restarted/expired. Copy current INFERENCE_URL from Colab cell 2.")
                    raise InferenceConnectionError(f"HTTP 404 Not Found at {endpoint}. Ensure Ngrok URL is active.")
                if attempt <= self.max_retries and err.code >= 500:
                    time.sleep(1.0)
                    continue
                raise InferenceHTTPError(f"HTTP Error {err.code}: {err.reason}")

            except urllib.error.URLError as err:
                latency_ms = int((time.time() - start_time) * 1000)
                print(f"[INFERENCE] request_id={request_id} ERROR=URLError reason='{err.reason}' latency_ms={latency_ms}")
                if attempt <= self.max_retries:
                    time.sleep(1.0)
                    continue
                raise InferenceConnectionError(f"Connection to inference endpoint failed: {err.reason}")

        raise InferenceConnectionError(f"InferenceClient failed after {self.max_retries + 1} attempts.")
