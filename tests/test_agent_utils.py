import pytest
from src.agent_loop import ReActAgent

def test_extract_code_success():
    text = "Here is some code:\n```python\nprint('Hello, world!')\n```\nIt works!"
    code = ReActAgent.extract_code(text)
    assert code == "print('Hello, world!')"

def test_extract_code_no_code():
    text = "Here is some text without code."
    code = ReActAgent.extract_code(text)
    assert code is None

def test_extract_code_multiple_blocks():
    text = "First block:\n```python\na = 1\n```\nSecond block:\n```python\nb = 2\n```"
    code = ReActAgent.extract_code(text)
    assert code == "a = 1"

def test_extract_code_empty_block():
    text = "Empty block:\n```python\n\n```"
    code = ReActAgent.extract_code(text)
    assert code == ""

def test_extract_code_missing_python_tag():
    text = "Missing python tag:\n```\nprint('Hello')\n```"
    code = ReActAgent.extract_code(text)
    assert code is None
