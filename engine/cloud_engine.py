"""
Cloud Engine for El GPT.
Provides ultra-fast, zero-hardware-load LLM streaming via:
1. Groq Cloud API (450 to 750+ tokens per second, 0% CPU/RAM load)
2. OpenRouter API (Access to free top-tier models: Nemotron 120B, Gemma 26B, Laguna S)
"""

import base64
import json
import os
import re
from typing import Dict, Generator, List, Optional
import urllib.request
import urllib.error

try:
    import requests
except ImportError:
    requests = None

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

    # ─── Free Image Studio (FLUX.1) ───
    "el-gpt-image-flux": {
        "provider": "image",
        "api_model": "flux",
        "name": "FLUX.1 Image Studio",
        "tag": "🎨 Free Image AI",
        "badge": "FLUX.1 Free",
        "description": "State-of-the-art text-to-image generator powered by FLUX.1. Create stunning art, photos, 3D renders & concept art for 100% free.",
        "max_tokens": 1024,
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
    "You are El GPT, an intelligent, helpful, and friendly AI assistant. "
    "Your tone is warm, thoughtful, natural, and distinctly human—never robotic, dry, or difficult to read.\n\n"
    "CRITICAL FORMATTING & READABILITY GUIDELINES:\n"
    "- Always format your answers for effortless reading, clarity, and scannability.\n"
    "- Use clean bullet points (`- `) and numbered lists whenever explaining concepts, listing steps, or breaking down points.\n"
    "- Use bold headings (`### `) to organize different sections.\n"
    "- Keep paragraphs short (2 to 3 sentences maximum) with clear line breaks between paragraphs.\n"
    "- Use **bold text** on key takeaways, terms, and important details so the user can quickly grasp the main ideas.\n"
    "- Never dump dense, unformatted walls of text.\n"
    "- When writing code, provide clean, modern, fully functional code inside markdown code blocks with the correct language tag.\n"
    "- When an image is attached, provide a rich, structured visual breakdown with bullet points describing objects, text, colors, and direct answers to the user's inquiry."
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


def find_env_var(target_name: str) -> Optional[str]:
    """Resolves an environment variable case-insensitively and ignoring underscores."""
    val = os.environ.get(target_name)
    if val and val.strip():
        return val.strip().strip("'\"")

    clean_target = target_name.lower().replace("_", "")
    for k, v in os.environ.items():
        if k.lower().replace("_", "") == clean_target and v and v.strip():
            return v.strip().strip("'\"")
    return None


def get_api_key(client_provided_key: Optional[str] = None, provider: str = "groq") -> Optional[str]:
    """
    Resolves API key for the requested provider ('groq' or 'openrouter').
    Checks:
    1. Client provided key (from browser localStorage)
    2. Environment variables with flexible case/naming variations
    3. Root .env file
    """
    load_env_file()

    if client_provided_key and client_provided_key.strip():
        k = client_provided_key.strip().strip("'\"")
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
        for var in ["OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPENROUTER"]:
            val = find_env_var(var)
            if val:
                if "," in val:
                    return val.split(",")[0].strip().strip("'\"")
                return val
        pool = find_env_var("OPENROUTER_API_KEYS")
        if pool:
            return pool.split(",")[0].strip().strip("'\"")
        return None

    # Groq provider
    for var in ["GROQ_API_KEY", "GROQ_KEY", "GROQ"]:
        val = find_env_var(var)
        if val:
            return val
    return None


def get_openrouter_keys_pool() -> List[str]:
    """Returns all available OpenRouter keys from environment to handle failover."""
    load_env_file()
    keys = []
    for var in ["OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPENROUTER"]:
        val = find_env_var(var)
        if val and val not in keys:
            keys.append(val)
    pool = find_env_var("OPENROUTER_API_KEYS")
    if pool:
        for k in pool.split(","):
            clean_k = k.strip().strip("'\"")
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

        # 1. Image Generation Route (Zero API Key Needed, 100% Free)
        if provider == "image" or model_id == "el-gpt-image-flux":
            from .image_engine import generate_image, clean_image_prompt
            last_prompt = messages[-1].get("content", "") if messages else "A futuristic neon dreamscape"
            clean = clean_image_prompt(last_prompt)
            img_result = generate_image(clean or last_prompt)
            img_url = img_result["image_url"]
            fallback_url = img_result.get("fallback_url", img_url)

            yield f"🎨 **Image Creator**:\n\n"
            yield f"> *\"{clean}\"*\n\n"
            yield f'<div class="image-bubble-container"><img src="{img_url}" alt="{clean}" class="generated-image" data-fallback="{fallback_url}" /><div class="image-actions-row"><button class="download-image-btn" data-url="{img_url}" data-prompt="{clean}">⬇️ Download Image</button></div></div>\n\n'
            return

        # 2. Universal /image command trigger from any model
        if messages:
            last_text = messages[-1].get("content", "").strip()
            if last_text.lower().startswith("/image ") or last_text.lower().startswith("/img "):
                from .image_engine import generate_image, clean_image_prompt
                prompt_text = re.sub(r"^/(image|img)\s+", "", last_text, flags=re.IGNORECASE).strip()
                clean = clean_image_prompt(prompt_text)
                img_result = generate_image(clean or prompt_text)
                img_url = img_result["image_url"]
                fallback_url = img_result.get("fallback_url", img_url)

                yield f"🎨 **Image Creator**:\n\n"
                yield f"> *\"{clean}\"*\n\n"
                yield f'<div class="image-bubble-container"><img src="{img_url}" alt="{clean}" class="generated-image" data-fallback="{fallback_url}" /><div class="image-actions-row"><button class="download-image-btn" data-url="{img_url}" data-prompt="{clean}">⬇️ Download Image</button></div></div>\n\n'
                return

        # Resolve primary key for text models
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

        # Check if any message contains an attached image for vision models
        has_images = any(bool(m.get("image")) for m in messages)
        api_model = model_info["api_model"]

        # If an image is attached, automatically ensure a vision-capable model is used
        if has_images:
            if provider == "groq":
                # Qwen 27B on Groq provides lightning-fast native vision
                api_model = "qwen/qwen3.8-27b"
            elif provider == "openrouter":
                api_model = "qwen/qwen3.8-27b:free"

        # Build payload messages with standard OpenAI/Groq multimodal schema
        formatted_messages = []
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            formatted_messages.append({"role": "system", "content": system_content})

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            img = m.get("image")
            if img:
                formatted_messages.append({
                    "role": role,
                    "content": [
                        {"type": "text", "text": content or "Please analyze and describe this image in detail with clear bullet points."},
                        {"type": "image_url", "image_url": {"url": img}}
                    ]
                })
            else:
                formatted_messages.append({"role": role, "content": content})

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
                post_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    endpoint,
                    data=post_data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    for raw_line in resp:
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8", errors="replace").strip()
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

            except urllib.error.HTTPError as http_err:
                status_code = http_err.code
                try:
                    err_body = http_err.read().decode("utf-8", errors="replace")
                    err_data = json.loads(err_body)
                    err_msg = err_data.get("error", {}).get("message", err_body)
                except Exception:
                    err_msg = str(http_err)

                # If 429 rate-limited and we have another key to try, failover
                if status_code == 429 and len(candidate_keys) > 1 and current_key != candidate_keys[-1]:
                    continue

                last_error = f"HTTP {status_code}: {err_msg}"
                if status_code == 401:
                    yield f"⚠️ **{provider.capitalize()} API Error (401 Unauthorized)**: Invalid API Key. Please verify your key in the Cloud Settings modal."
                elif status_code == 429:
                    yield f"⚠️ **{provider.capitalize()} Rate Limit (429)**: The free upstream model is busy. Please wait a few seconds and try again, or switch to **Nemotron 3 Super 120B**."
                else:
                    yield f"⚠️ **{provider.capitalize()} API Error ({status_code})**: {err_msg}"
                return

            except (urllib.error.URLError, TimeoutError) as net_err:
                last_error = f"Network Connection Error: {net_err}"
            except Exception as e:
                last_error = str(e)

        if not success:
            yield f"\n⚠️ **Cloud Request Error**: {last_error or 'Unable to complete streaming request'}"


cloud_engine = CloudEngine()
