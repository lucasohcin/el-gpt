"""
Cloud Engine for El GPT.
Provides ultra-fast, zero-hardware-load LLM streaming via:
1. Groq Cloud API (450 to 750+ tokens per second, 0% CPU/RAM load)
2. OpenRouter API (Access to free top-tier models: Nemotron 120B, Gemma 26B, Laguna S)
"""

import json
import os
import re
from typing import Dict, Generator, List, Optional
import requests

from .memory import memory_engine

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Comprehensive Model Mapping for Cloud Providers
CLOUD_MODELS = {
    # ─── Groq Models (Ultra-Fast 450-750 tps) ───
    "el-gpt-cloud-120b": {
        "provider": "groq",
        "api_model": "openai/gpt-oss-120b",
        "name": "El GPT Cloud 120B",
        "tag": "⚡ 450 tps Groq",
        "badge": "Groq 120B",
        "description": "Massive 120B parameter powerhouse on Groq. Blazing speed, elite coding, deep reasoning, mathematics & full-stack web.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-qwen-27b": {
        "provider": "groq",
        "api_model": "qwen/qwen3.8-27b",
        "name": "El GPT Qwen 3.8 27B",
        "tag": "⚡ 400 tps Groq",
        "badge": "Qwen 27B",
        "description": "State-of-the-art coding and multilingual engine on Groq. Superior HTML5/CSS3/JS frontend generation & math.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-20b": {
        "provider": "groq",
        "api_model": "openai/gpt-oss-20b",
        "name": "El GPT Cloud Fast 20B",
        "tag": "⚡ 750 tps Groq",
        "badge": "Fast 20B",
        "description": "Ultra-high speed 20B model on Groq. Near-instantaneous streaming (750+ tokens/sec) for rapid conversation.",
        "max_tokens": 4096,
    },

    # ─── OpenRouter Free Models ───
    "el-gpt-or-nemotron-120b": {
        "provider": "openrouter",
        "api_model": "nvidia/nemotron-3-super-120b-a12b:free",
        "name": "Nemotron 3 Super 120B",
        "tag": "🌐 OpenRouter Free",
        "badge": "Nvidia 120B",
        "description": "Nvidia Nemotron 3 Super 120B on OpenRouter. High-powered enterprise reasoning and software engineering with free tier access.",
        "max_tokens": 4096,
    },
    "el-gpt-or-gemma-26b": {
        "provider": "openrouter",
        "api_model": "google/gemma-4-26b-a4b-it:free",
        "name": "Gemma 4 26B (Free)",
        "tag": "🌐 OpenRouter Free",
        "badge": "Gemma 26B",
        "description": "Google Gemma 26B on OpenRouter. Versatile, fast instruction-tuned language model for dialogue and code generation.",
        "max_tokens": 4096,
    },
    "el-gpt-or-laguna-s": {
        "provider": "openrouter",
        "api_model": "poolside/laguna-s-2.1:free",
        "name": "Poolside Laguna S 2.1",
        "tag": "🌐 OpenRouter Free",
        "badge": "Laguna S",
        "description": "Poolside Laguna S 2.1 on OpenRouter. Specialized reasoning and developer assistant with free tier access.",
        "max_tokens": 4096,
    },

    # ─── Backwards-Compatible Aliases ───
    "el-gpt-cloud-llama-70b": {
        "provider": "groq",
        "api_model": "openai/gpt-oss-120b",
        "name": "El GPT Cloud 120B",
        "tag": "⚡ 450 tps Groq",
        "badge": "Groq 120B",
        "description": "Massive 120B parameter powerhouse on Groq.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-deepseek-r1": {
        "provider": "groq",
        "api_model": "openai/gpt-oss-120b",
        "name": "El GPT Reasoning 120B",
        "tag": "⚡ Reasoning Groq",
        "badge": "120B Pro",
        "description": "Deep reasoning engine on Groq.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-qwen-32b": {
        "provider": "groq",
        "api_model": "qwen/qwen3.8-27b",
        "name": "El GPT Qwen 3.8 27B",
        "tag": "⚡ 400 tps Groq",
        "badge": "Qwen 27B",
        "description": "State-of-the-art coding and multilingual engine on Groq.",
        "max_tokens": 4096,
    },
    "el-gpt-cloud-llama-8b": {
        "provider": "groq",
        "api_model": "openai/gpt-oss-20b",
        "name": "El GPT Cloud Fast 20B",
        "tag": "⚡ 750 tps Groq",
        "badge": "Fast 20B",
        "description": "Ultra-high speed 20B model on Groq.",
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


def get_api_key(client_provided_key: Optional[str] = None, provider: str = "groq") -> Optional[str]:
    """
    Resolves API key for the requested provider ('groq' or 'openrouter').
    Checks:
    1. Client provided key (from localStorage)
    2. Environment variable (GROQ_API_KEY or OPENROUTER_API_KEY)
    3. Root .env file
    """
    load_env_file()

    if client_provided_key and client_provided_key.strip():
        k = client_provided_key.strip()
        # Auto-detect provider if key prefix is unmistakable
        if k.startswith("sk-or-") and provider == "openrouter":
            return k
        if k.startswith("gsk_") and provider == "groq":
            return k
        # If no strict prefix clash, use client key
        if provider == "openrouter" and not k.startswith("gsk_"):
            return k
        if provider == "groq" and not k.startswith("sk-or-"):
            return k

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            keys_pool = os.environ.get("OPENROUTER_API_KEYS", "").strip()
            if keys_pool:
                key = keys_pool.split(",")[0].strip()
        return key if key else None

    # Groq provider
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return key if key else None


def get_openrouter_keys_pool() -> List[str]:
    """Returns all available OpenRouter keys from environment to handle failover."""
    load_env_file()
    keys = []
    primary = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if primary:
        keys.append(primary)
    pool = os.environ.get("OPENROUTER_API_KEYS", "").strip()
    if pool:
        for k in pool.split(","):
            clean_k = k.strip()
            if clean_k and clean_k not in keys:
                keys.append(clean_k)
    return keys


def save_api_key_to_env(key: str, provider: str = "groq") -> bool:
    """Saves or updates API keys in the root .env file."""
    clean_key = key.strip()
    if not clean_key:
        return False

    env_var_name = "OPENROUTER_API_KEY" if (clean_key.startswith("sk-or-") or provider == "openrouter") else "GROQ_API_KEY"

    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(root_dir, ".env")
        lines = []
        key_found = False
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith(f"{env_var_name}="):
                        lines.append(f"{env_var_name}={clean_key}\n")
                        key_found = True
                    else:
                        lines.append(line)
        if not key_found:
            lines.append(f"{env_var_name}={clean_key}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        os.environ[env_var_name] = clean_key
        return True
    except Exception as e:
        print(f"[CloudEngine] Error saving API key: {e}")
        return False


class CloudEngine:
    def __init__(self):
        self.groq_endpoint = GROQ_ENDPOINT
        self.openrouter_endpoint = OPENROUTER_ENDPOINT

    def is_cloud_model(self, model_id: str) -> bool:
        return model_id in CLOUD_MODELS

    def get_models_metadata(self) -> List[Dict]:
        models = []
        # Return distinct models (skip backwards compatible aliases in menu)
        seen_names = set()
        for mid, info in CLOUD_MODELS.items():
            if info["name"] in seen_names and mid.startswith("el-gpt-cloud-llama"):
                continue
            seen_names.add(info["name"])
            models.append({
                "id": mid,
                "name": info["name"],
                "tag": info["tag"],
                "badge": info["badge"],
                "provider": info["provider"],
                "parameters": "Cloud Host",
                "description": info["description"],
                "status": f"Ready ({info['provider'].capitalize()} ⚡)",
            })
        return models

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model_id: str = "el-gpt-cloud-120b",
        api_key: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        model_info = CLOUD_MODELS.get(model_id, CLOUD_MODELS["el-gpt-cloud-120b"])
        provider = model_info.get("provider", "groq")

        # Resolve primary key
        resolved_key = get_api_key(api_key, provider=provider)

        if not resolved_key:
            if provider == "openrouter":
                yield (
                    "### 🌐 OpenRouter API Key Needed\n\n"
                    f"You selected **{model_info['name']}**, which runs via **OpenRouter**!\n\n"
                    "To start chatting:\n"
                    "1. Get your free key at **[openrouter.ai/keys](https://openrouter.ai/keys)**.\n"
                    "2. Click the **⚡ Cloud** button in the top navigation bar to save your OpenRouter key.\n\n"
                    "*(Or add `OPENROUTER_API_KEY=sk-or-v1-...` in your server `.env` or Vercel Environment Variables).*"
                )
            else:
                yield (
                    "### ⚡ Groq Cloud API Key Needed\n\n"
                    f"You selected **{model_info['name']}**, which runs on **Groq Cloud** at **450+ tokens/sec**!\n\n"
                    "To start chatting for free:\n"
                    "1. Get your free key at **[console.groq.com](https://console.groq.com)** (takes 30 seconds, no credit card required).\n"
                    "2. Click the **⚡ Cloud** button in the top navigation bar to save your key.\n\n"
                    "*(Or add `GROQ_API_KEY=gsk_...` in your server `.env` or Vercel Environment Variables).*"
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

        api_model = model_info["api_model"]
        endpoint = self.openrouter_endpoint if provider == "openrouter" else self.groq_endpoint

        # Candidate keys to try (failover for rate limits)
        candidate_keys = [resolved_key]
        if provider == "openrouter":
            for pool_key in get_openrouter_keys_pool():
                if pool_key not in candidate_keys:
                    candidate_keys.append(pool_key)

        success = False
        last_error = ""

        for current_key in candidate_keys:
            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json",
                "User-Agent": "El-GPT-Engine/1.0",
            }
            if provider == "openrouter":
                headers["HTTP-Referer"] = "https://el-gpt.vercel.app"
                headers["X-Title"] = "El GPT"

            payload = {
                "model": api_model,
                "messages": formatted_messages,
                "stream": True,
                "temperature": max(0.1, min(temperature, 1.2)),
                "max_tokens": min(max_new_tokens, model_info.get("max_tokens", 4096)),
                "top_p": top_p,
            }

            try:
                with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=60) as resp:
                    if resp.status_code != 200:
                        try:
                            err_data = resp.json()
                            err_msg = err_data.get("error", {}).get("message", resp.text)
                        except Exception:
                            err_msg = resp.text

                        # If 429 rate-limited and we have another key to try, failover
                        if resp.status_code == 429 and len(candidate_keys) > 1 and current_key != candidate_keys[-1]:
                            continue

                        last_error = f"HTTP {resp.status_code}: {err_msg}"
                        if resp.status_code == 401:
                            yield f"⚠️ **{provider.capitalize()} API Error (401 Unauthorized)**: Invalid API Key. Please verify your key in the Cloud Settings modal."
                        elif resp.status_code == 429:
                            yield f"⚠️ **{provider.capitalize()} Rate Limit (429)**: The free upstream model is busy. Please wait a few seconds and try again, or switch to **Nemotron 3 Super 120B**."
                        else:
                            yield f"⚠️ **{provider.capitalize()} API Error ({resp.status_code})**: {err_msg}"
                        return

                    # Stream tokens
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8")
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                success = True
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

                    if success:
                        return

            except requests.exceptions.Timeout:
                last_error = "Connection Timeout"
            except requests.exceptions.ConnectionError:
                last_error = "Network Connection Error"
            except Exception as e:
                last_error = str(e)

        if not success:
            yield f"\n⚠️ **Cloud Request Error**: {last_error or 'Unable to complete streaming request'}"


cloud_engine = CloudEngine()
