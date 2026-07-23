import openai
import re
import subprocess
import os
import sys
import argparse
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env, если он существует
load_dotenv()

MODEL_NAME = "gpt-4o"

SYSTEM_PROMPT = """
Ты — Senior AI Researcher и эксперт по Mechanistic Interpretability.
Твоя цель: исследовать веса, активации и поведение предоставленной PyTorch-модели.

ПРАВИЛА ИССЛЕДОВАНИЯ:
1. ДОСТУП К МОДЕЛИ: Исследуемая модель и токенизатор доступны через библиотеку `transformers` или `transformer_lens`.
   Для начала загрузи небольшую модель, например `gpt2`, чтобы понять базовую структуру.
2. ЗАПРЕТ НА СЫРЫЕ ВЕСА: НИКОГДА не выводи в консоль полные тензоры (`print(model.weights)`).
   Используй агрегацию: выводи `.shape`, `.mean()`, `.std()`, или гистограммы в виде текста.
   Переполнение вывода приведет к провалу миссии.
3. ИЗОЛЯЦИЯ СКРИПТОВ: Каждый раз ты должен писать ПОЛНЫЙ, исполняемый Python-скрипт с нужными импортами.
   Состояние между запусками скриптов не сохраняется. Сохраняй промежуточные данные в локальные файлы (например, `.json` или `.pt`), если они понадобятся на следующем шаге.
4. ОТЛАДКА: Если скрипт падает с ошибкой, я пришлю тебе Traceback. Проанализируй его (особенно внимательно следи за несовпадением размерностей тензоров — `size mismatch`) и перепиши код.

ФОРМАТ ОТВЕТА:
Сначала напиши свои рассуждения. Затем предоставь код строго внутри блока ```python ... ```.
За один раз ты можешь написать только один блок кода.
"""

class ReActAgent:
    def __init__(self, client, model_name=MODEL_NAME, system_prompt=SYSTEM_PROMPT):
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages = []

    @staticmethod
    def extract_code(text):
        """Извлекает Python-код из ответа LLM."""
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def set_memory_limit():
        import resource
        # Ограничиваем память до 4 ГБ (4 * 1024 * 1024 * 1024 байт)
        max_mem_bytes = 4 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))

    @staticmethod
    def execute_script(code, filename="agent_workspace/experiment.py"):
        """Сохраняет код в файл и безопасно запускает его в подпроцессе с ограничением памяти."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        # Запуск в изолированном процессе
        try:
            result = subprocess.run(
                ["unshare", "-r", "-n", sys.executable, filename],
                capture_output=True,
                text=True,
                timeout=120, # Ограничение времени выполнения (защита от бесконечных циклов)
                preexec_fn=ReActAgent.set_memory_limit # Ограничение памяти
            )
            output = result.stdout
            if result.stderr:
                output += f"\nERRORS:\n{result.stderr}"
            return output if output.strip() else "Скрипт выполнен успешно, но ничего не вывел в консоль."
        except subprocess.TimeoutExpired:
            return "Ошибка: Превышено время ожидания выполнения (Timeout)."
        except Exception as e:
            return f"Системная ошибка при запуске: {str(e)}"

    def run(self, user_goal, max_steps=10):
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_goal}
        ]

        for step in range(max_steps):
            print(f"\n--- ШАГ {step + 1} ---")

            try:
                # 1. Запрос к LLM (Мозг)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    temperature=0.2
                )
                agent_reply = response.choices[0].message.content
                print(f"АГЕНТ:\n{agent_reply}\n")
                self.messages.append({"role": "assistant", "content": agent_reply})

                # 2. Поиск и выполнение кода (Действие)
                code = self.extract_code(agent_reply)

                if code:
                    print(">> Выполняю код агента...")
                    execution_result = self.execute_script(code)

                    # Усекаем слишком длинный вывод, чтобы не забить контекст
                    if len(execution_result) > 2000:
                        execution_result = execution_result[:2000] + "\n...[ВЫВОД ОБРЕЗАН ИЗ-ЗА ДЛИНЫ]..."

                    print(f"<< РЕЗУЛЬТАТ:\n{execution_result}")

                    # 3. Передача результатов обратно агенту (Наблюдение)
                    self.messages.append({
                        "role": "user",
                        "content": f"Результат выполнения скрипта:\n```text\n{execution_result}\n```\nЧто делаем дальше?"
                    })
                else:
                    # Если кода нет, агент считает, что задача выполнена, или задает вопрос
                    print(">> Агент не предоставил код. Остановка цикла или ожидание ответа.")
                    break
            except Exception as e:
                print(f"Ошибка при работе с API LLM: {str(e)}")
                break

def run_agent_loop(user_goal, client, model_name=MODEL_NAME, max_steps=10):
    agent = ReActAgent(client=client, model_name=model_name)
    agent.run(user_goal, max_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск Mechanistic Interpretability Агента")
    parser.add_argument("--goal", type=str, default="Загрузи модель GPT-2 из transformers. Выведи структуру её слоев и найди, какой размер имеет матрица весов в первом слое Feed Forward Network (MLP).", help="Цель для агента")
    parser.add_argument("--api-key", type=str, help="OpenAI API ключ (или другой поддерживаемый). Также можно задать через OPENAI_API_KEY в .env")
    parser.add_argument("--base-url", type=str, help="Базовый URL API (для локальных моделей, например vLLM/Ollama). Также можно задать через OPENAI_BASE_URL в .env")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Имя модели (по умолчанию: {MODEL_NAME})")
    parser.add_argument("--max-steps", type=int, default=10, help="Максимальное количество шагов агента")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        print("Внимание: API ключ не задан. Используется фиктивный 'dummy_key'. Установите переменную OPENAI_API_KEY в .env файле или передайте через --api-key.")
        api_key = "dummy_key"

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = openai.OpenAI(**client_kwargs)

    run_agent_loop(args.goal, client, model_name=args.model, max_steps=args.max_steps)
