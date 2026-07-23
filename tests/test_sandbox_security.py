import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent_loop import ReActAgent

def test_sandbox_network_is_blocked():
    code = """
import urllib.request
try:
    urllib.request.urlopen('http://google.com', timeout=3)
    print("Network is accessible")
except Exception as e:
    print(f"Network is blocked: {e}")
"""
    output = ReActAgent.execute_script(code, "agent_workspace/test_net.py")

    # We expect the network to be blocked
    assert "Network is blocked" in output
    assert "Network is accessible" not in output
