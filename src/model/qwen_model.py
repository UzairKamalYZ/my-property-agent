import ollama
from src.config import Config

class QwenModel:
    """Wrapper for local Qwen model using Ollama."""

    def __init__(self, model_name=Config.QWEN_MODEL_NAME):
        self.model = model_name

    def chat(self, messages, stream=False):
        """Send messages to the Qwen model."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                stream=stream
            )
            if stream:
                return self._handle_stream(response)
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"Error during model chat: {e}")
            return ""

    def _handle_stream(self, response):
        """Handle streaming responses."""
        full_response = ""
        for chunk in response:
            content = chunk["message"]["content"]
            full_response += content
            yield content

    def close(self):
        """Close the model and release resources."""
        pass  # No-op for now, but good practice to have
