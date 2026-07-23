import os
import json
import pytest
from unittest.mock import MagicMock
from src.agent_loop import ReActAgent

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Mock the API response
    response_mock = MagicMock()
    response_mock.choices = [MagicMock()]
    response_mock.choices[0].message.content = "Тестовый ответ агента. ```python\nprint('Test')\n```"
    client.chat.completions.create.return_value = response_mock
    return client

def test_save_and_load_state(mock_client, tmp_path):
    # 1. Create a new agent and run it for 1 step to generate some state
    save_path = tmp_path / "session.json"
    agent1 = ReActAgent(client=mock_client)

    # We set max_steps=1 so it only does one cycle
    agent1.run("Тестовая цель 1", max_steps=1, save_path=str(save_path))

    # Check that file was created
    assert save_path.exists()

    # Check that state was saved properly
    with open(save_path, 'r', encoding='utf-8') as f:
        saved_messages = json.load(f)

    assert len(saved_messages) > 2 # System + User + Assistant + Tool
    assert saved_messages[0]["role"] == "system"
    assert saved_messages[1]["role"] == "user"
    assert saved_messages[1]["content"] == "Тестовая цель 1"

    # 2. Create a second agent and load the state
    agent2 = ReActAgent(client=mock_client)
    agent2.load_state(str(save_path))

    # Verify that the loaded messages match the saved ones
    assert agent2.messages == saved_messages

    # 3. Run the second agent with a new goal
    agent2.run("Тестовая цель 2", max_steps=1, save_path=str(save_path))

    # Check that the new goal was appended correctly
    assert agent2.messages[-4]["role"] == "user" # Previous tool output
    assert agent2.messages[-3]["role"] == "user" # New goal
    assert agent2.messages[-3]["content"] == "Тестовая цель 2"
    assert agent2.messages[-2]["role"] == "assistant"
