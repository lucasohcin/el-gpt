"""
Inference & Text Generation Engine for El GPT 1.0.
Supports real-time streaming, temperature, top-k, top-p, and repetition penalties.
"""

from typing import Dict, Generator, List, Optional
import torch
import torch.nn.functional as F

try:
    from .model import ElGPTModel
    from .tokenizer import ElGPTTokenizer
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.model import ElGPTModel
    from engine.tokenizer import ElGPTTokenizer


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_top_p_top_k(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    recent_tokens: Optional[List[int]] = None,
) -> int:
    """Samples next token with temperature, top-k, top-p, and repetition penalty."""
    logits = logits.clone()

    # Repetition penalty
    if recent_tokens and repetition_penalty != 1.0:
        for tid in set(recent_tokens):
            if logits[0, tid] > 0:
                logits[0, tid] /= repetition_penalty
            else:
                logits[0, tid] *= repetition_penalty

    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature

    # Top-K
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    # Top-P (Nucleus)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def build_prompt_tokens(messages: List[Dict[str, str]], tokenizer: ElGPTTokenizer) -> List[int]:
    """Builds token sequence matching training format exactly."""
    tokens = [tokenizer.bos_id]
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        prefix = f"{tokenizer.system_token}\n"
        tokens.extend(tokenizer.encode(prefix))
        tokens.extend(tokenizer.encode(
            "You are El GPT 1.0, an intelligent, helpful, and concise AI engine specializing in mathematics, software coding, and daily conversational chatting."
        ))
        tokens.append(tokenizer.eos_id)
        tokens.extend(tokenizer.encode("\n"))

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            role_token = tokenizer.system_token
        elif role == "assistant":
            role_token = tokenizer.assistant_token
        else:
            role_token = tokenizer.user_token

        prefix = f"{role_token}\n"
        tokens.extend(tokenizer.encode(prefix))
        tokens.extend(tokenizer.encode(content))
        tokens.append(tokenizer.eos_id)
        tokens.extend(tokenizer.encode("\n"))

    # Append assistant trigger
    tokens.extend(tokenizer.encode(f"{tokenizer.assistant_token}\n"))
    return tokens


@torch.no_grad()
def generate_stream(
    model: ElGPTModel,
    tokenizer: ElGPTTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 350,
    temperature: float = 0.3,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: Optional[torch.device] = None,
) -> Generator[str, None, None]:
    """
    Yields decoded text chunks as tokens are generated in real-time.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    prompt_tokens = build_prompt_tokens(messages, tokenizer)

    # Crop to max_seq_len - max_new_tokens if too long
    max_input_len = model.config.max_seq_len - max_new_tokens
    if len(prompt_tokens) > max_input_len:
        prompt_tokens = prompt_tokens[-max_input_len:]

    curr_input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    stop_tokens = {tokenizer.eos_id, tokenizer.user_id, tokenizer.system_id}

    generated_tokens: List[int] = []

    for _ in range(max_new_tokens):
        # Forward pass on current sequence
        logits, _, _ = model(curr_input_ids)
        next_token_logits = logits[:, -1, :]

        recent = generated_tokens[-20:] if generated_tokens else None
        next_token = sample_top_p_top_k(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            recent_tokens=recent,
        )

        if next_token in stop_tokens:
            break

        generated_tokens.append(next_token)
        chunk = tokenizer.decode([next_token], skip_special=True)
        if chunk:
            yield chunk

        # Append next token to input
        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        curr_input_ids = torch.cat([curr_input_ids, next_tensor], dim=1)

        # Ensure we don't exceed max context window
        if curr_input_ids.size(1) >= model.config.max_seq_len:
            break
