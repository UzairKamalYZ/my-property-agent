import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import json
from unittest.mock import patch
from src.memory_manager import MemoryManager

@pytest.fixture
def temp_memory_file(tmp_path):
    return tmp_path / "memory.json"

@pytest.fixture
def temp_urls_file(tmp_path):
    return tmp_path / "urls.txt"

@pytest.fixture
def memory_manager(temp_memory_file, temp_urls_file):
    return MemoryManager(memory_file=temp_memory_file, urls_file=temp_urls_file)

def test_load_memory_file_not_found(memory_manager):
    assert memory_manager.load_memory() == [{"role": "system", "content": "You are a helpful and concise assistant with memory."}]

def test_load_memory_file_found(memory_manager, temp_memory_file):
    memory_data = [{"role": "system", "content": "test"}]
    with open(temp_memory_file, "w") as f:
        json.dump(memory_data, f)
    assert memory_manager.load_memory() == memory_data

def test_save_memory(memory_manager, temp_memory_file):
    memory_data = [{"role": "system", "content": "test"}]
    memory_manager.save_memory(memory_data)
    with open(temp_memory_file, "r") as f:
        assert json.load(f) == memory_data

