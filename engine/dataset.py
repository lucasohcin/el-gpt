"""
Dataset loader and formatting for El GPT 1.0 training.
Formats multi-turn conversations and masks loss on user prompts so the model
focuses gradient updates on assistant responses.
"""

import json
import os
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .tokenizer import ElGPTTokenizer
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.tokenizer import ElGPTTokenizer


def format_conversation_tokens(messages: List[Dict[str, str]], tokenizer: ElGPTTokenizer, max_seq_len: int) -> Dict[str, torch.Tensor]:
    """
    Encodes conversation turns into token IDs with assistant response loss masking.
    """
    input_tokens: List[int] = [tokenizer.bos_id]
    target_tokens: List[int] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            role_token = tokenizer.system_token
            is_assistant = False
        elif role == "user":
            role_token = tokenizer.user_token
            is_assistant = False
        else:
            role_token = tokenizer.assistant_token
            is_assistant = True

        # Segment: "<|role|>\n" + content + "<|eos|>\n"
        prefix = f"{role_token}\n"
        prefix_ids = tokenizer.encode(prefix)
        content_ids = tokenizer.encode(content)
        suffix_ids = [tokenizer.eos_id] + tokenizer.encode("\n")

        # Prefix is prompt context — mask out from target loss
        for token_id in prefix_ids:
            input_tokens.append(token_id)
            target_tokens.append(-100)

        # Assistant content and EOS are target predictions
        for token_id in (content_ids + suffix_ids):
            input_tokens.append(token_id)
            if is_assistant:
                target_tokens.append(token_id)
            else:
                target_tokens.append(-100)

    # Truncate to max_seq_len + 1 (for input and shifted target)
    if len(input_tokens) > max_seq_len + 1:
        input_tokens = input_tokens[: max_seq_len + 1]
        target_tokens = target_tokens[: max_seq_len]
    else:
        target_tokens = target_tokens[: len(input_tokens) - 1]

    # Shift for causal prediction: input is [0 : N-1], target is [1 : N]
    inp = input_tokens[:-1]
    tgt = target_tokens

    # Pad to max_seq_len
    pad_len = max_seq_len - len(inp)
    if pad_len > 0:
        inp = inp + [tokenizer.pad_id] * pad_len
        tgt = tgt + [-100] * pad_len

    return {
        "input_ids": torch.tensor(inp, dtype=torch.long),
        "targets": torch.tensor(tgt, dtype=torch.long),
    }


class ElGPTChatDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: ElGPTTokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, torch.Tensor]] = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    messages = record.get("messages", [])
                    if messages:
                        tensor_dict = format_conversation_tokens(messages, tokenizer, max_seq_len)
                        self.samples.append(tensor_dict)
                except Exception as e:
                    print(f"[Dataset Warning] Skipped malformed line: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def create_dataloader(data_path: str, tokenizer: ElGPTTokenizer, batch_size: int = 4, max_seq_len: int = 512, shuffle: bool = True) -> DataLoader:
    dataset = ElGPTChatDataset(data_path, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
