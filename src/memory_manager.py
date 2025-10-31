import os
import json
from src.config import Config

class MemoryManager:
    """Manages the conversation memory for the agent."""

    def __init__(self, memory_file=Config.MEMORY_FILE, urls_file=Config.URLS_FILE):
        self.memory_file = memory_file
        self.urls_file = urls_file

    def load_memory(self):
        """Load conversation history from memory file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading memory file: {e}")
        return [{"role": "system", "content": "You are a helpful and concise assistant with memory."}]

    def save_memory(self, memory):
        """Save conversation history to memory file."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(memory, f, indent=2)
        except IOError as e:
            print(f"Error saving memory file: {e}")

    
