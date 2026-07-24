import pytest
from src.agent_loop import ReActAgent, run_planner_agent_loop
import io
import sys
import json
from unittest.mock import patch

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

    def create(self, model, messages, temperature):
        self.call_count += 1

        # В `run_planner_agent_loop` planner.messages содержит только пользовательский промпт до первого вызова
        # Но мы можем посмотреть на системный промпт агента через вызов, либо определить по содержимому last_msg

        last_msg = messages[-1]["content"] if messages else ""

        # Planner получает глобальную цель первым запросом
        is_planner = "Глобальная цель:" in last_msg
        # Reviewer получает "Оригинальная цель:"
        is_reviewer = "Оригинальная цель:" in last_msg

        if is_planner:
            return MockResponse([MockChoice(MockMessage('[\n  "Подзадача 1: Сделать что-то",\n  "Подзадача 2: Сделать что-то еще"\n]'))])
        elif is_reviewer:
            return MockResponse([MockChoice(MockMessage("ОДОБРЕНО. Все отлично."))])
        else:
            return MockResponse([MockChoice(MockMessage("Я выполнил задачу.\n```python\nprint('Готово')\n```"))])

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockClient:
    def __init__(self):
        self.chat = MockChat()


def test_planner_multi_agent_interaction():
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        mock_client = MockClient()

        # Мокаем execute_script, чтобы не запускать песочницу
        with patch.object(ReActAgent, 'execute_script', return_value="Выполнение замокано"):
            run_planner_agent_loop("Глобальная цель: сделать все круто", client=mock_client, model_name="test-model", max_steps=1, max_reviews=1)

        output = captured_output.getvalue()

        assert "ПЛАНИРОВЩИК ЗАДАЧ РАБОТАЕТ" in output
        assert "Задач в плане: 2" in output
        assert "=== ВЫПОЛНЕНИЕ ПОДЗАДАЧИ 1 ИЗ 2 ===" in output
        assert "=== ВЫПОЛНЕНИЕ ПОДЗАДАЧИ 2 ИЗ 2 ===" in output
        assert "RESEARCHER РАБОТАЕТ" in output
        assert "REVIEWER ОЦЕНИВАЕТ" in output
        assert "REVIEWER ОДОБРИЛ РЕЗУЛЬТАТ. ЗАВЕРШЕНИЕ" in output

    finally:
        sys.stdout = sys.__stdout__
