"""
El GPT 1.8 Ultra Engine (~1.54 Billion Parameters).
Backed by Qwen2.5-1.5B-Instruct (1,543,714,816 parameters, pre-trained on 18T tokens).
Runs natively on Apple Silicon Metal Performance Shaders (MPS) in float16 precision.

Modes:
- Ultra: Maximum reasoning depth, memory injection, full-stack web, mathematics, and systems engineering.
- Fast: Low-latency streaming with focused top_k and lower temperature.
"""

import threading
import time
from typing import Dict, Generator, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .memory import memory_engine


MODEL_ID_1B = "Qwen/Qwen2.5-1.5B-Instruct"

BASE_SYSTEM_PROMPT = (
    "You are El GPT 1.8 Ultra, a state-of-the-art 1 Billion+ parameter AI engine "
    "running natively on Apple Silicon hardware acceleration. You possess elite capabilities in "
    "computer science, modern HTML5/CSS3/JavaScript web development, Python, systems programming, "
    "multi-step mathematics, and intelligent, nuanced conversation like ChatGPT and Gemini. "
    "Always provide clean, modern, well-commented code, rigorous step-by-step reasoning, and adhere "
    "closely to the user's stored preferences."
)


class ElGPT18Engine:
    def __init__(self, model_id: str = MODEL_ID_1B):
        self.model_id = model_id
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self._lock = threading.Lock()

    def load_model(self):
        with self._lock:
            if self.is_loaded:
                return
            print(f"[El GPT 1.8] Loading 1.5B model ({self.model_id}) on {self.device} in float16...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
            self.is_loaded = True
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"[El GPT 1.8] Successfully loaded! Parameters: {param_count:,}")

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        mode: str = "ultra",  # "ultra" or "fast"
        max_new_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        if not self.is_loaded:
            self.load_model()

        # Check if latest user message contains a memory to store
        if messages:
            last_user_msg = messages[-1].get("content", "")
            implicit_mem = memory_engine.extract_implicit_memory(last_user_msg)
            if implicit_mem:
                memory_engine.add_memory(implicit_mem)

        # Assemble system prompt with user preferences
        system_content = BASE_SYSTEM_PROMPT
        memory_context = memory_engine.format_memory_prompt()
        system_content += memory_context

        formatted_messages = []
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            formatted_messages.append({"role": "system", "content": system_content})

        for m in messages:
            formatted_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        prompt_text = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        if mode == "fast":
            temp = 0.2 if temperature is None else temperature
            top_k = 25
            max_tokens = min(max_new_tokens, 512)
            do_sample = True if temp > 0.05 else False
        else:
            # Ultra mode: deep reasoning
            temp = 0.6 if temperature is None else temperature
            top_k = 50
            max_tokens = max_new_tokens
            do_sample = True if temp > 0.05 else False

        stop_ids = [self.tokenizer.eos_token_id]
        for extra_id in [151645, 151643]:
            if extra_id not in stop_ids:
                stop_ids.append(extra_id)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=max(temp, 0.1),
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=stop_ids,
            repetition_penalty=1.15,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk

        thread.join()


el_gpt_1_8 = ElGPT18Engine()
