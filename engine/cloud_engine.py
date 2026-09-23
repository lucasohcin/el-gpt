"""
Cloud Engine for El GPT.
Provides ultra-fast, zero-hardware-load LLM streaming via Groq Cloud API.
Generates 300 to 750+ tokens per second with 0% CPU/RAM load on the host machine.
"""

import json
import os
import re
from typing import Dict, Generator, List, Optional
import requests

from .memory import memory_engine

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Model Mapping: El GPT Cloud IDs -> Groq Model Identifiers
CLOUD_MODELS = {
    "el-gpt-cloud-120b": {
        "groq_model": "openai/gpt-oss-120b",
        "name": "El GPT Cloud 120B",
        "tag": "⚡ 450 tps Cloud",
        "badge": "Cloud 120B",
        "description": "Massive 120B parameter powerhouse in the cloud. Blazing speed, elite coding, deep reasoning, mathematics & full-stack web.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-qwen-27b": {
        "groq_model": "qwen/qwen3.8-27b",
        "name": "El GPT Qwen 3.8 27B",
        "tag": "⚡ 400 tps Cloud",
        "badge": "Qwen 27B",
        "description": "State-of-the-art coding and multilingual engine. Superior HTML5/CSS3/JS frontend generation & math.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-20b": {
        "groq_model": "openai/gpt-oss-20b",
        "name": "El GPT Cloud Fast 20B",
        "tag": "⚡ 750 tps Ultra-Speed",
        "badge": "Fast 20B",
        "description": "Ultra-high speed 20B model. Near-instantaneous streaming (750+ tokens/sec) for rapid conversation.",
        "max_tokens": 4096,
    },
    # Backwards-compatible aliases
    "el-gpt-cloud-llama-70b": {
        "groq_model": "openai/gpt-oss-120b",
        "name": "El GPT Cloud 120B",
        "tag": "⚡ 450 tps Cloud",
        "badge": "Cloud 120B",
        "description": "Massive 120B parameter powerhouse in the cloud. Blazing speed, elite coding, mathematics & systems engineering.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-deepseek-r1": {
        "groq_model": "openai/gpt-oss-120b",
        "name": "El GPT Reasoning 120B",
        "tag": "⚡ Reasoning Cloud",
        "badge": "120B Pro",
        "description": "Deep reasoning engine. Step-by-step chain-of-thought proofs, advanced logic, algorithms & coding.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-qwen-32b": {
        "groq_model": "qwen/qwen3.8-27b",
        "name": "El GPT Qwen 3.8 27B",
        "tag": "⚡ 400 tps Cloud",
        "badge": "Qwen 27B",
        "description": "State-of-the-art coding and multilingual engine. Superior HTML/CSS/JS frontend generation & math.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-llama-8b": {
        "groq_model": "openai/gpt-oss-20b",
        "name": "El GPT Cloud Fast 20B",
        "tag": "⚡ 750 tps Ultra-Speed",
        "badge": "Fast 20B",
        "description": "Ultra-high speed 20B model. Near-instantaneous streaming (750+ tokens/sec) for rapid conversation.",
        "max_tokens": 4096,
    },
}

BASE_SYSTEM_PROMPT = (
    "You are El GPT, an elite AI assistant and software engineering engine running on ultra-fast cloud infrastructure. "
    "You have state-of-the-art expertise in modern full-stack web development (HTML5, CSS3, modern JavaScript/TypeScript), "
    "Python, algorithms, mathematics, and natural intelligent conversation. "
    "Always provide clean, modern, well-commented code, step-by-step reasoning where applicable, and follow the user's preferences."
)


def load_env_file():
    """Reads .env file into os.environ if present."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            print(f"[CloudEngine] Could not load .env: {e}")


# Preload .env on import
load_env_file()


def get_api_key(client_provided_key: Optional[str] = None) -> Optional[str]:
    """
    Resolves the Groq API key in order of priority:
    1. Key explicitly passed in the request (from browser localStorage)
    2. Environment variable GROQ_API_KEY
    3. .env file in project root
    """
    if client_provided_key and client_provided_key.strip():
        return client_provided_key.strip()

    load_env_file()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return key if key else None


def save_api_key_to_env(key: str) -> bool:
    """Saves or updates GROQ_API_KEY in the root .env file."""
    clean_key = key.strip()
    if not clean_key:
        return False
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(root_dir, ".env")
        lines = []
        key_found = False
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GROQ_API_KEY="):
                        lines.append(f"GROQ_API_KEY={clean_key}\n")
                        key_found = True
                    else:
                        lines.append(line)
        if not key_found:
            lines.append(f"GROQ_API_KEY={clean_key}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        os.environ["GROQ_API_KEY"] = clean_key
        return True
    except Exception as e:
        print(f"[CloudEngine] Error saving API key: {e}")
        return False


class CloudEngine:
    def __init__(self):
        self.endpoint = GROQ_ENDPOINT

    def is_cloud_model(self, model_id: str) -> bool:
        return model_id in CLOUD_MODELS

    def get_models_metadata(self) -> List[Dict]:
        models = []
        for mid, info in CLOUD_MODELS.items():
            models.append({
                "id": mid,
                "name": info["name"],
                "tag": info["tag"],
                "badge": info["badge"],
                "parameters": "Cloud Host",
                "description": info["description"],
                "status": "Ready (Cloud ⚡)",
            })
        return models

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model_id: str = "el-gpt-cloud-llama-70b",
        api_key: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        resolved_key = get_api_key(api_key)

        if not resolved_key:
            yield (
                "### ⚡ Groq Cloud API Key Needed\n\n"
                "You selected an **El GPT Cloud Model**, which runs at **400+ tokens/sec** with **0% Mac CPU usage**!\n\n"
                "To start chatting for free:\n"
                "1. Get your free key at **[console.groq.com](https://console.groq.com)** (takes 30 seconds, no credit card required).\n"
                "2. Click the **⚡ Cloud Key** button in the top navigation bar to save your key in this browser.\n\n"
                "*(Or create a `.env` file with `GROQ_API_KEY=gsk_...` on your server).* \n\n"
                "Once entered, you will experience instant, blazing-fast AI streaming!"
            )
            return

        # Check if latest user message contains a memory to store
        if messages:
            last_user_msg = messages[-1].get("content", "")
            implicit_mem = memory_engine.extract_implicit_memory(last_user_msg)
            if implicit_mem:
                memory_engine.add_memory(implicit_mem)

        # Assemble system prompt with memory preferences
        system_content = BASE_SYSTEM_PROMPT
        memory_context = memory_engine.format_memory_prompt()
        if memory_context:
            system_content += memory_context

        # Build payload messages
        formatted_messages = []
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            formatted_messages.append({"role": "system", "content": system_content})

        for m in messages:
            formatted_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        model_info = CLOUD_MODELS.get(model_id, CLOUD_MODELS["el-gpt-cloud-120b"])
        groq_model = model_info["groq_model"]

        headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
            "User-Agent": "El-GPT-Engine/1.0",
        }

        payload = {
            "model": groq_model,
            "messages": formatted_messages,
            "stream": True,
            "temperature": max(0.1, min(temperature, 1.2)),
            "max_tokens": min(max_new_tokens, model_info.get("max_tokens", 4096)),
            "top_p": top_p,
        }

        try:
            with requests.post(self.endpoint, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get("error", {}).get("message", resp.text)
                    except Exception:
                        err_msg = resp.text

                    if resp.status_code == 401:
                        yield f"⚠️ **Groq API Error (401 Unauthorized)**: Invalid API Key. Please verify your key in the Cloud Settings modal."
                    elif resp.status_code == 429:
                        yield f"⚠️ **Groq API Error (429 Rate Limit)**: High traffic or rate limit reached. Please wait a moment and retry."
                    else:
                        yield f"⚠️ **Groq API Error ({resp.status_code})**: {err_msg}"
                    return

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.Timeout:
            yield "\n⚠️ **Connection Timeout**: The Groq Cloud API timed out. Please check your internet connection."
        except requests.exceptions.ConnectionError:
            yield "\n⚠️ **Network Error**: Unable to reach api.groq.com. Please check your network connection."
        except Exception as e:
            yield f"\n⚠️ **Unexpected Cloud Error**: {str(e)}"


cloud_engine = CloudEngine()
