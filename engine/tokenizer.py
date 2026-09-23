"""
El GPT 1.0 Custom Byte-Level Tokenizer
Supports:
- Chat markup special tokens (<|system|>, <|user|>, <|assistant|>, <|eos|>, etc.)
- 256 byte-level fallbacks so any character/emoji is lossless
- Multi-character subwords for Python, JavaScript, Math, and conversational English
- JSON serialization/deserialization
"""

import json
import os
import re
from typing import Dict, List, Optional


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
]

DEFAULT_SUBWORDS = [
    # Common whitespace & indentation
    "  ", "    ", "        ", "\n", "\n\n", "\t",
    # Programming keywords & symbols
    "def ", "class ", "return ", "import ", "from ", "if ", "elif ", "else:", "for ", "while ",
    "in ", "not ", "and ", "or ", "is ", "None", "True", "False", "self", "print(", "len(",
    "function ", "const ", "let ", "var ", "async ", "await ", "try:", "except:", "finally:",
    "console.log", "typeof ", "instanceof", "=>", "==", "!=", "<=", ">=", "+=", "-=", "*=",
    "[]", "{}", "()", '""', "''", "://", "http", "https", "github", "json", "html", "css",
    # Math symbols & keywords
    "\\frac", "\\sqrt", "\\sum", "\\int", "\\pi", "\\infty", "\\theta", "\\alpha", "\\beta",
    "\\approx", "\\times", "\\div", "\\pm", "\\cdot", "\\partial", "dx", "dy", "dt", "f(x)",
    "x^2", "x^3", "+", "-", "*", "/", "=", "<", ">", "^", "%", "!", "|", "$", "$$",
    "Step 1", "Step 2", "Step 3", "Step 4", "Solution:", "Answer:", "Explanation:",
    # Common English words and punctuation
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on",
    "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
    "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make",
    "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "El GPT", "El GPT 1.0", "Hello", "Hi", "Sure", "Certainly", "Here", "is", "code", "math",
    ", ", ". ", "? ", "! ", ": ", "; ", " - ", " -> ", '": "', '", '
]


DEFAULT_VOCAB_PATH = "data/tokenizer_vocab.json"


class ElGPTTokenizer:
    def __init__(self, vocab_file: Optional[str] = None):
        self.special_tokens = list(SPECIAL_TOKENS)
        self.pad_token = "<|pad|>"
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        self.system_token = "<|system|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        target_file = vocab_file or DEFAULT_VOCAB_PATH
        if os.path.exists(target_file):
            self.load(target_file)
        else:
            self._build_default_vocab()
            try:
                self.save(target_file)
            except Exception:
                pass

    def _build_default_vocab(self):
        self.token_to_id.clear()
        self.id_to_token.clear()

        # 1. Special tokens first
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # 2. 256 raw byte tokens (e.g. <0x00> .. <0xFF>) for universal fallback
        for b in range(256):
            token = f"<0x{b:02X}>"
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # 3. High-frequency subwords & keywords
        for subword in DEFAULT_SUBWORDS:
            if subword not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[subword] = idx
                self.id_to_token[idx] = subword

        # 4. Extract words & tokens from training curriculum if available
        train_path = "data/train.jsonl"
        if os.path.exists(train_path):
            try:
                with open(train_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        for m in rec.get("messages", []):
                            content = m.get("content", "")
                            # Extract words and markdown/code patterns
                            tokens_found = re.findall(r"\w+|[^\w\s]", content)
                            for t in tokens_found:
                                if t not in self.token_to_id:
                                    idx = len(self.token_to_id)
                                    self.token_to_id[t] = idx
                                    self.id_to_token[idx] = t
            except Exception as e:
                print(f"[Tokenizer Warning] Could not parse train dataset: {e}")

        self._compile_regex()

    def _compile_regex(self):
        # Sort subwords by length descending for greedy matching
        sorted_tokens = sorted(
            [k for k in self.token_to_id.keys() if not k.startswith("<0x")],
            key=lambda x: len(x),
            reverse=True,
        )
        escaped = [re.escape(t) for t in sorted_tokens]
        self.pattern = re.compile("|".join(escaped) if escaped else r".")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def user_id(self) -> int:
        return self.token_to_id[self.user_token]

    @property
    def assistant_id(self) -> int:
        return self.token_to_id[self.assistant_token]

    @property
    def system_id(self) -> int:
        return self.token_to_id[self.system_token]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_id)

        pos = 0
        n = len(text)

        while pos < n:
            match = self.pattern.match(text, pos)
            if match:
                token_str = match.group(0)
                tokens.append(self.token_to_id[token_str])
                pos = match.end()
            else:
                # Byte fallback for arbitrary character/bytes
                char = text[pos]
                for b in char.encode("utf-8"):
                    token_str = f"<0x{b:02X}>"
                    tokens.append(self.token_to_id.get(token_str, self.pad_id))
                pos += 1

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        byte_buffer = bytearray()
        result_chunks: List[str] = []

        def flush_bytes():
            if byte_buffer:
                result_chunks.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in ids:
            if token_id not in self.id_to_token:
                continue
            token_str = self.id_to_token[token_id]

            if token_str in self.special_tokens:
                flush_bytes()
                if not skip_special:
                    result_chunks.append(token_str)
                continue

            if token_str.startswith("<0x") and token_str.endswith(">") and len(token_str) == 6:
                try:
                    b = int(token_str[3:5], 16)
                    byte_buffer.append(b)
                    continue
                except ValueError:
                    pass

            flush_bytes()
            result_chunks.append(token_str)

        flush_bytes()
        return "".join(result_chunks)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.special_tokens = data.get("special_tokens", SPECIAL_TOKENS)
        self._compile_regex()


if __name__ == "__main__":
    tok = ElGPTTokenizer()
    print(f"[ElGPTTokenizer] Vocab size: {tok.vocab_size}")
    sample = "Hello! I am El GPT 1.0. Let's solve $\\int 2x dx = x^2 + C$ and write `def solve(): return 42`."
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"Sample length: {len(sample)} chars -> {len(encoded)} tokens")
    assert sample == decoded, "Decoded text did not match original!"
    print("Lossless roundtrip test passed successfully!")
