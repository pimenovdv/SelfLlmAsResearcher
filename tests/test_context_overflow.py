import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent_loop import ReActAgent

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

class MockCompletions:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def create(self, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return MockResponse(resp)
        return MockResponse("")

class MockChat:
    def __init__(self, responses):
        self.completions = MockCompletions(responses)

class MockClient:
    def __init__(self, responses):
        self.chat = MockChat(responses)

def test_context_overflow():
    # Агент должен сгенерировать скрипт, который выводит >2000 символов
    script_content = 'print("A" * 3000)'
    agent_reply = f"Test\n```python\n{script_content}\n```"

    mock_client = MockClient([agent_reply])
    agent = ReActAgent(client=mock_client)

    agent.run("Test goal", max_steps=1)

    # Проверяем, что в истории сообщений есть результат с усеченным выводом
    # Последнее сообщение - это результат выполнения (так как max_steps=1, то он остановится или сделает 1 шаг)
    last_message = agent.messages[-1]["content"]
    assert "...[ВЫВОД ОБРЕЗАН ИЗ-ЗА ДЛИНЫ]..." in last_message

    # Проверяем, что длина усеченного вывода примерно равна 2000 + длина строки-заглушки + обертки
    # Обертка: "Результат выполнения скрипта:\n```text\n{execution_result}\n```\nЧто делаем дальше?"
    assert len(last_message) < 2200
