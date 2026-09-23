"""
User Long-Term Memory & Preferences Engine for El GPT 1.5.
Stores, updates, and formats persistent user preferences in data/user_memory.json.
Allows El GPT 1.5 to remember user preferences across sessions (just like ChatGPT).
"""

import json
import os
import re
from typing import List, Optional


MEMORY_FILE = "data/user_memory.json"

DEFAULT_MEMORIES = [
    "User prefers clean, modern code formatting with clear comments.",
    "User prefers step-by-step mathematical reasoning.",
    "User prefers modern dark-mode responsive designs for web development.",
]


class MemoryEngine:
    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_MEMORIES, f, indent=2)

    def get_memories(self) -> List[str]:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return list(DEFAULT_MEMORIES)

    def add_memory(self, memory: str) -> bool:
        clean = memory.strip()
        if not clean:
            return False
        memories = self.get_memories()
        if clean not in memories:
            memories.append(clean)
            self._save(memories)
        return True

    def delete_memory(self, index: int) -> bool:
        memories = self.get_memories()
        if 0 <= index < len(memories):
            memories.pop(index)
            self._save(memories)
            return True
        return False

    def clear_memories(self):
        self._save([])

    def _save(self, memories: List[str]):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)

    def extract_implicit_memory(self, user_message: str) -> Optional[str]:
        """Detects if user is asking the model to remember something."""
        text = user_message.strip()
        patterns = [
            r"(?:please\s+)?remember\s+that\s+(.+)",
            r"(?:from\s+now\s+on\s*,?\s*)?(?:always\s+)?remember\s+(.+)",
            r"my\s+name\s+is\s+([A-Za-z]+)",
            r"i\s+prefer\s+(.+)",
            r"i\s+love\s+(.+)",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().rstrip(".!?")
                if "my name is" in pat:
                    return f"User's name is {match.group(1).capitalize()}."
                elif "i prefer" in pat:
                    return f"User prefers {extracted}."
                elif "i love" in pat:
                    return f"User loves {extracted}."
                return f"User stated: {extracted}."
        return None

    def format_memory_prompt(self) -> str:
        memories = self.get_memories()
        if not memories:
            return ""
        items = "\n".join(f"- {m}" for m in memories)
        return (
            "\n\n[USER PROFILE & SAVED PREFERENCES]\n"
            "Here are verified facts and preferences about the user you are chatting with:\n"
            f"{items}\n"
            "Use these facts to personalize your answers, remember their preferences, and answer questions about them."
        )


memory_engine = MemoryEngine()
