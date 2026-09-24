"""
Free Image Generation Engine for El GPT.
Powered by FLUX.1 (100% Free, Zero API Key Required).
Generates high-resolution 1024x1024 photorealistic and artistic images.
"""

import random
import re
import urllib.parse
from typing import Dict, Optional


def clean_image_prompt(raw_text: str) -> str:
    """Cleans up chat prefixes like 'draw me', 'generate an image of', etc."""
    text = raw_text.strip()
    # Strip common leading command patterns
    patterns = [
        r"^/image\s*",
        r"^/img\s*",
        r"^(please\s+)?(draw|generate|create|paint|render|make)\s+(an?\s+image\s+of\s+|a\s+picture\s+of\s+|a\s+photo\s+of\s+|an?\s+art\s+of\s+|an?\s+)?",
        r"^(can\s+you\s+)?(draw|generate|make)\s+(me\s+)?(an?\s+)?",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()
    return text or raw_text.strip()


def enhance_prompt(prompt: str) -> str:
    """Adds subtle quality boosters to short prompts for stunning FLUX rendering."""
    clean = clean_image_prompt(prompt)
    if len(clean.split()) <= 4:
        return f"{clean}, 8k resolution, cinematic lighting, highly detailed, photorealistic masterpiece"
    return clean


def generate_image(prompt: str, width: int = 1024, height: int = 1024, model: str = "flux") -> Dict[str, any]:
    """
    Generates a high-quality image URL via FLUX.1.
    No API key required, 100% free.
    """
    enhanced = enhance_prompt(prompt)
    encoded = urllib.parse.quote(enhanced)
    seed = random.randint(10000, 99999999)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&model={model}&nologo=true&seed={seed}"

    return {
        "success": True,
        "image_url": url,
        "clean_prompt": clean_image_prompt(prompt),
        "enhanced_prompt": enhanced,
        "seed": seed,
        "model": model,
        "width": width,
        "height": height,
    }
