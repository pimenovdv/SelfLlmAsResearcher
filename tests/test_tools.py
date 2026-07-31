import os
import pytest
from src.tools import read_file, write_file, bash_command

def test_write_and_read_file(tmp_path):
    # Test writing to a file
    test_file = tmp_path / "test.txt"
    content = "Hello, World!\nThis is a test."

    result = write_file(str(test_file), content)
    assert result == f"Successfully wrote to {test_file}"
    assert test_file.exists()

    # Test reading from a file
    read_content = read_file(str(test_file))
    assert read_content == content

def test_read_nonexistent_file():
    # Test reading a file that doesn't exist
    result = read_file("nonexistent_file.txt")
    assert "Error reading file" in result

def test_write_file_error(monkeypatch):
    # Test error handling when writing a file
    def mock_open(*args, **kwargs):
        raise IOError("Mock IOError")

    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("builtins.open", mock_open)
    result = write_file("dummy_path.txt", "content")
    assert "Error writing to file dummy_path.txt: Mock IOError" in result

def test_bash_command_success():
    # Test a successful bash command
    result = bash_command("echo 'Hello, World!'")
    assert result.strip() == "Hello, World!"

def test_bash_command_error():
    # Test a bash command that produces an error
    result = bash_command("ls nonexistent_directory")
    assert "ERRORS:" in result
    assert "No such file or directory" in result

def test_bash_command_no_output():
    # Test a bash command that has no output
    result = bash_command("true")
    assert result == "Command executed successfully with no output."

def test_bash_command_timeout(monkeypatch):
    import subprocess
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="sleep 100", timeout=60)

    monkeypatch.setattr(subprocess, "run", mock_run)
    result = bash_command("sleep 100")
    assert result == "Error: Command execution timed out."
