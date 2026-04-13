"""
Agent Loop Implementation for CSE 476 Final Project

The agent loop is designed to use at least 8 inference-time reasoning algorithms
or techniques.
"""


import os, json, textwrap, re, time
import requests
import dotenv

dotenv.load_dotenv()

API_KEY  = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")  
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")  

cot_prompt = "You are an analytical reasoning assistant. Use step-by-step reasoning to solve the question, do not include any explanations or reasoning in your response, only the final answer. Keep your response as short as possible, do not describe your reasoning at all."


def chain_of_thought(question: str, 
                    system: str = cot_prompt,
                    model: str = MODEL,
                    temperature: float = 0.0,
                    timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": question}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text
            #return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
           
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}


def agent_loop(question: str) -> str:
   
    answer = chain_of_thought(question)
    return f"{answer}"