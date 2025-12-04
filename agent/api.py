import requests

API_KEY  = "cse476"
API_BASE = "http://10.4.58.53:41701/v1"
MODEL    = "bens_model"

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60):
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

        # Success
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "ok": True,
                "text": text,
                "raw": data,
                "status": 200,
                "error": None
            }

        # Failure with server response
        return {
            "ok": False,
            "text": None,
            "raw": None,
            "status": resp.status_code,
            "error": resp.text
        }

    except Exception as e:
        # Network error or timeout
        return {
            "ok": False,
            "text": None,
            "raw": None,
            "status": -1,
            "error": str(e)
        }
