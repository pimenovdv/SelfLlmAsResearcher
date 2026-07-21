import openai
import re
import subprocess
import os
import sys

# Инициализация клиента (замените на ваш API-ключ или локальный эндпоинт, например Ollama/vLLM)
# Для работы примера необходимо установить переменную окружения OPENAI_API_KEY
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy_key"))
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

def extract_code(text):
    """Извлекает Python-код из ответа LLM."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else None

def execute_script(code, filename="agent_workspace/experiment.py"):
    """Сохраняет код в файл и безопасно запускает его в подпроцессе."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    # Запуск в изолированном процессе
    try:
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=120 # Ограничение времени выполнения (защита от бесконечных циклов)
        )
        output = result.stdout
        if result.stderr:
            output += f"\nERRORS:\n{result.stderr}"
        return output if output.strip() else "Скрипт выполнен успешно, но ничего не вывел в консоль."
    except subprocess.TimeoutExpired:
        return "Ошибка: Превышено время ожидания выполнения (Timeout)."
    except Exception as e:
        return f"Системная ошибка при запуске: {str(e)}"

def run_agent_loop(user_goal, max_steps=10):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal}
    ]

    for step in range(max_steps):
        print(f"\n--- ШАГ {step + 1} ---")

        try:
            # 1. Запрос к LLM (Мозг)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2
            )
            agent_reply = response.choices[0].message.content
            print(f"АГЕНТ:\n{agent_reply}\n")
            messages.append({"role": "assistant", "content": agent_reply})

            # 2. Поиск и выполнение кода (Действие)
            code = extract_code(agent_reply)

            if code:
                print(">> Выполняю код агента...")
                execution_result = execute_script(code)

                # Усекаем слишком длинный вывод, чтобы не забить контекст
                if len(execution_result) > 2000:
                    execution_result = execution_result[:2000] + "\n...[ВЫВОД ОБРЕЗАН ИЗ-ЗА ДЛИНЫ]..."

                print(f"<< РЕЗУЛЬТАТ:\n{execution_result}")

                # 3. Передача результатов обратно агенту (Наблюдение)
                messages.append({
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

if __name__ == "__main__":
    goal = "Загрузи модель GPT-2 из transformers. Выведи структуру её слоев и найди, какой размер имеет матрица весов в первом слое Feed Forward Network (MLP)."
    run_agent_loop(goal)
