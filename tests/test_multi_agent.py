import pytest
from src.agent_loop import ReActAgent, run_multi_agent_loop
import io
import sys

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, message):
        self.message = message

class MockResponse:
    def __init__(self, choices):
        self.choices = choices

class MockCompletions:
    def __init__(self):
        self.call_count = 0
        self.is_researcher = True # Пытаемся понять по роли, но мы будем ориентироваться на чередование в тесте

    def create(self, model, messages, temperature):
        self.call_count += 1

        # Если последний промпт содержит "REVIEWER ОЦЕНИВАЕТ" или это промпт ревьювера
        last_msg = messages[-1]["content"]
        is_reviewer = "Оригинальная цель:" in last_msg

        if not is_reviewer:
            # Это Researcher
            if self.call_count == 1:
                return MockResponse([MockChoice(MockMessage("Я написал плохой код.\n```python\nprint(1)\n```"))])
            else:
                return MockResponse([MockChoice(MockMessage("Я исправил код.\n```python\nprint('Исправлено')\n```"))])
        else:
            # Это Reviewer
            if "print(1)" in last_msg:
                return MockResponse([MockChoice(MockMessage("ОТКЛОНЕНО. Вывод плохой. Исправь." ))])
            else:
                return MockResponse([MockChoice(MockMessage("ОДОБРЕНО. Все отлично." ))])

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockClient:
    def __init__(self):
        self.chat = MockChat()

from unittest.mock import patch

def test_multi_agent_interaction():
    # Перехватываем stdout, чтобы проверить вывод
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        mock_client = MockClient()

        # Мокаем execute_script, чтобы не запускать песочницу
        with patch.object(ReActAgent, 'execute_script', return_value="Выполнение замокано"):
            # Ставим max_steps=1, чтобы исследователь делал 1 ход за итерацию
            run_multi_agent_loop("Цель: сделать хорошо", client=mock_client, model_name="test-model", max_steps=1, max_reviews=3)

        output = captured_output.getvalue()

        assert "RESEARCHER РАБОТАЕТ" in output
        assert "REVIEWER ОЦЕНИВАЕТ" in output
        assert "ОТПРАВКА КОММЕНТАРИЕВ REVIEWER К RESEARCHER" in output
        assert "REVIEWER ОДОБРИЛ РЕЗУЛЬТАТ. ЗАВЕРШЕНИЕ" in output

    finally:
        sys.stdout = sys.__stdout__
