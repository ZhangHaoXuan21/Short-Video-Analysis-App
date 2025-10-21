import os
import pickle
from typing import Dict, List


class MemoryManager:
    def __init__(self, persist: bool = True, filename: str = "memory_store.pkl"):
        # Structure: { user_id: { session_id: [ {role, content}, ... ] } }
        self.memory_store: Dict[str, Dict[str, List[Dict[str, str]]]] = {}

        # Persistence settings
        self.persist = persist
        self.filename = os.path.join(os.path.dirname(__file__), filename)

        # Load persisted memory if available
        if self.persist:
            self.load_memory()

    # ---- Persistence ----

    def save_memory(self):
        """Save memory to a pickle file."""
        try:
            with open(self.filename, "wb") as f:
                pickle.dump(self.memory_store, f)
        except Exception as e:
            print(f"[MemoryManager] Warning: Failed to save memory -> {e}")

    def load_memory(self):
        """Load memory from a pickle file if it exists."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "rb") as f:
                    self.memory_store = pickle.load(f)
            except Exception as e:
                print(f"[MemoryManager] Warning: Failed to load memory -> {e}")
                self.memory_store = {}

    # ---- User & Session Management ----

    def add_chat_user(self, user_id: str):
        if user_id not in self.memory_store:
            self.memory_store[user_id] = {}
            if self.persist:
                self.save_memory()

    def add_chat_session(self, user_id: str, session_id: str):
        self.add_chat_user(user_id)
        if session_id not in self.memory_store[user_id]:
            self.memory_store[user_id][session_id] = []
            if self.persist:
                self.save_memory()

    # ---- Message Handling ----

    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        self.add_chat_session(user_id, session_id)
        self.memory_store[user_id][session_id].append({"role": role, "content": content})
        if self.persist:
            self.save_memory()

    def get_history(self, user_id: str, session_id: str) -> List[Dict[str, str]]:
        return self.memory_store.get(user_id, {}).get(session_id, [])

    def get_context(self, user_id: str, session_id: str, top_k: int | None = None) -> str:
        history = self.get_history(user_id, session_id)
        if not history:
            return ""
        if top_k is not None:
            history = history[-top_k:]
        return "\n".join([f"{m['role']}: {m['content']}" for m in history])

    # ---- Cleanup ----

    def clear_session(self, user_id: str, session_id: str):
        if user_id in self.memory_store and session_id in self.memory_store[user_id]:
            self.memory_store[user_id][session_id] = []
            if self.persist:
                self.save_memory()

    def remove_session(self, user_id: str, session_id: str):
        if user_id in self.memory_store:
            self.memory_store[user_id].pop(session_id, None)
            if self.persist:
                self.save_memory()

    def remove_user(self, user_id: str):
        self.memory_store.pop(user_id, None)
        if self.persist:
            self.save_memory()

    # ---- Utility ----

    def list_users(self) -> List[str]:
        return list(self.memory_store.keys())

    def list_sessions(self, user_id: str) -> List[str]:
        return list(self.memory_store.get(user_id, {}).keys())
