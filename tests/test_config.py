import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch
import importlib
from src import config

def test_config_loading():
    with patch.dict(os.environ, {"QWEN_MODEL_NAME": "test_model", "MEMORY_FILE": "test_memory.json"}):
        importlib.reload(config)
        assert config.Config.LLM_MODEL_NAME == "test_model"
        assert config.Config.MEMORY_FILE == "test_memory.json"

def test_config_defaults():
    with patch.dict(os.environ, {}, clear=True):
        importlib.reload(config)
        assert config.Config.LLM_MODEL_NAME == "qwen3:1.7b"
        assert config.Config.MEMORY_FILE == "memory.json"
