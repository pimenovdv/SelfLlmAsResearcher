import openai
import re
import subprocess
import os
import sys
import argparse
import json
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

PLANNER_SYSTEM_PROMPT = """
Ты — AI Planner и Архитектор сложных задач в области Mechanistic Interpretability.
Твоя цель: принять глобальную задачу пользователя и разбить её на логические, последовательные шаги (подзадачи), которые будут выполнены агентом-исследователем (Researcher).

ПРАВИЛА ПЛАНИРОВАНИЯ:
1. Задачи должны быть конкретными и выполнимыми за 1-2 запуска скрипта.
2. В каждой задаче указывай, что именно нужно сделать и какие результаты получить.
3. Формат вывода: ТОЛЬКО валидный JSON массив строк-задач. Никакого дополнительного текста до или после JSON.

Пример вывода:
[
  "Загрузить модель gpt2 и вывести её архитектуру, проверив форму первой матрицы весов MLP.",
  "Написать скрипт для извлечения активаций из MLP слоя при подаче на вход текста 'Hello world'."
]
"""

REVIEWER_SYSTEM_PROMPT = """
Ты — AI Reviewer и эксперт по Mechanistic Interpretability.
Твоя цель: оценивать скрипты и результаты экспериментов, выполненные Researcher агентом.

ПРАВИЛА ОЦЕНКИ:
1. Оценивай корректность Python кода.
2. Оценивай достижение исходной цели пользователя.
3. Проверяй, не выводит ли скрипт сырые веса (контекст должен быть агрегированным).
4. Ты можешь написать Python скрипт (в блоке ```python ... ```), чтобы запустить дополнительные проверки и ассерты. Если ты напишешь код, он будет выполнен в изолированной песочнице, и ты получишь результаты, прежде чем выносить финальный вердикт.

ФОРМАТ ОТВЕТА:
Если результаты эксперимента полностью отвечают на задачу пользователя и код написан корректно (в том числе после твоих проверок),
твой ответ должен обязательно содержать фразу "ОДОБРЕНО".
В противном случае укажи на ошибки или предложи улучшения, чтобы Researcher их исправил. Не пиши код за него, только давай комментарии.
"""

class ReActAgent:
    def __init__(self, client, model_name=MODEL_NAME, system_prompt=SYSTEM_PROMPT):
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages = []

    def save_state(self, filepath):
        """Сохраняет текущую историю сообщений в JSON файл."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            print(f"Состояние успешно сохранено в {filepath}")
        except Exception as e:
            print(f"Ошибка при сохранении состояния: {e}")

    def load_state(self, filepath):
        """Загружает историю сообщений из JSON файла."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
                print(f"Состояние успешно загружено из {filepath}")
            else:
                print(f"Файл {filepath} не найден. Начинаем новую сессию.")
        except Exception as e:
            print(f"Ошибка при загрузке состояния: {e}")

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

    def run(self, user_goal, max_steps=10, save_path=None):
        if not self.messages:
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_goal}
            ]
        elif user_goal:
             # Если загрузили состояние, но есть новая цель, и последняя цель отличается, добавляем её
             has_recent_matching_goal = any(m["role"] == "user" and m["content"] == user_goal for m in self.messages[-3:])
             if not has_recent_matching_goal:
                 self.messages.append({"role": "user", "content": user_goal})

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

                if save_path:
                    self.save_state(save_path)
            except Exception as e:
                print(f"Ошибка при работе с API LLM: {str(e)}")
                break

def run_agent_loop(user_goal, client, model_name=MODEL_NAME, max_steps=10, resume_path=None, save_path=None):
    agent = ReActAgent(client=client, model_name=model_name)
    if resume_path:
        agent.load_state(resume_path)
    agent.run(user_goal, max_steps, save_path)

def run_planner_agent_loop(user_goal, client, model_name=MODEL_NAME, max_steps=10, max_reviews=3):
    print(">> ПЛАНИРОВЩИК ЗАДАЧ РАБОТАЕТ <<")
    planner = ReActAgent(client=client, model_name=model_name, system_prompt=PLANNER_SYSTEM_PROMPT)
    planner.messages.append({"role": "user", "content": user_goal})

    try:
        response = planner.client.chat.completions.create(
            model=planner.model_name,
            messages=planner.messages,
            temperature=0.2
        )
        planner_reply = response.choices[0].message.content
        print(f"PLANNER:\n{planner_reply}\n")

        # Попытка найти JSON массив в ответе планировщика
        match = re.search(r"\[.*\]", planner_reply, re.DOTALL)
        if match:
            tasks = json.loads(match.group(0))
        else:
            print("Ошибка: Планировщик не вернул JSON массив. Запуск без планировщика.")
            run_multi_agent_loop(user_goal, client, model_name, max_steps, max_reviews)
            return

        print(f"Задач в плане: {len(tasks)}")
        for i, task in enumerate(tasks):
            print(f"\n==========================================")
            print(f"=== ВЫПОЛНЕНИЕ ПОДЗАДАЧИ {i+1} ИЗ {len(tasks)} ===")
            print(f"==========================================\n")
            print(f"Подзадача: {task}\n")
            run_multi_agent_loop(task, client, model_name, max_steps, max_reviews)

    except Exception as e:
        print(f"Ошибка при работе Planner API: {str(e)}")

def run_multi_agent_loop(user_goal, client, model_name=MODEL_NAME, max_steps=10, max_reviews=3):
    researcher = ReActAgent(client=client, model_name=model_name, system_prompt=SYSTEM_PROMPT)
    reviewer = ReActAgent(client=client, model_name=model_name, system_prompt=REVIEWER_SYSTEM_PROMPT)

    current_goal = user_goal

    for review_step in range(max_reviews):
        print(f"\n==========================================")
        print(f"--- ИТЕРАЦИЯ MULTI-AGENT {review_step + 1} ---")
        print(f"==========================================\n")

        print(">> RESEARCHER РАБОТАЕТ <<")
        # Выполняем цикл Researcher
        researcher.run(current_goal, max_steps=max_steps)

        if not researcher.messages:
            print("Ошибка: Researcher не дал ответа.")
            break

        last_researcher_reply = researcher.messages[-1]["content"]
        if researcher.messages[-1]["role"] == "user":
            # Если последним был ответ среды, найдем последний ответ ассистента
            for msg in reversed(researcher.messages):
                if msg["role"] == "assistant":
                    last_researcher_reply = msg["content"]
                    break

        # Передаем контекст Reviewer
        reviewer_prompt = (
            f"Оригинальная цель: {user_goal}\n"
            f"Последние действия и результаты Researcher:\n"
            f"```\n{last_researcher_reply}\n```\n"
            f"Если в Researcher были ошибки (Traceback), учти их. "
            f"Если всё выполнено верно, напиши 'ОДОБРЕНО'."
        )

        print(">> REVIEWER ОЦЕНИВАЕТ <<")
        reviewer.messages.append({"role": "user", "content": reviewer_prompt})

        reviewer_approved = False
        try:
            # Даем Reviewer'у до 3 попыток на запуск проверочных скриптов
            for _ in range(3):
                response = reviewer.client.chat.completions.create(
                    model=reviewer.model_name,
                    messages=reviewer.messages,
                    temperature=0.2
                )
                reviewer_reply = response.choices[0].message.content
                reviewer.messages.append({"role": "assistant", "content": reviewer_reply})
                print(f"REVIEWER:\n{reviewer_reply}\n")

                reviewer_code = reviewer.extract_code(reviewer_reply)
                if reviewer_code:
                    print(">> Выполняю проверочный код Reviewer'а...")
                    execution_result = reviewer.execute_script(reviewer_code, filename="agent_workspace/reviewer_check.py")
                    if len(execution_result) > 2000:
                        execution_result = execution_result[:2000] + "\n...[ВЫВОД ОБРЕЗАН ИЗ-ЗА ДЛИНЫ]..."
                    print(f"<< РЕЗУЛЬТАТ ПРОВЕРКИ:\n{execution_result}")
                    reviewer.messages.append({
                        "role": "user",
                        "content": f"Результат выполнения твоего проверочного скрипта:\n```text\n{execution_result}\n```\nВынеси вердикт ('ОДОБРЕНО' или комментарии для исправления)."
                    })
                else:
                    if "ОДОБРЕНО" in reviewer_reply.upper():
                        reviewer_approved = True
                        print(">> REVIEWER ОДОБРИЛ РЕЗУЛЬТАТ. ЗАВЕРШЕНИЕ <<")
                    else:
                        print(">> ОТПРАВКА КОММЕНТАРИЕВ REVIEWER К RESEARCHER <<")
                        current_goal = f"Reviewer оставил комментарии к твоей работе. Пожалуйста, исправь ошибки:\n{reviewer_reply}"
                    break

            if reviewer_approved:
                break

        except Exception as e:
            print(f"Ошибка при работе Reviewer API: {str(e)}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск Mechanistic Interpretability Агента")
    parser.add_argument("--goal", type=str, default="Загрузи модель GPT-2 из transformers. Выведи структуру её слоев и найди, какой размер имеет матрица весов в первом слое Feed Forward Network (MLP).", help="Цель для агента")
    parser.add_argument("--api-key", type=str, help="OpenAI API ключ (или другой поддерживаемый). Также можно задать через OPENAI_API_KEY в .env")
    parser.add_argument("--base-url", type=str, help="Базовый URL API (для локальных моделей, например vLLM/Ollama). Также можно задать через OPENAI_BASE_URL в .env")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Имя модели (по умолчанию: {MODEL_NAME})")
    parser.add_argument("--max-steps", type=int, default=10, help="Максимальное количество шагов агента")
    parser.add_argument("--resume", type=str, help="Путь к JSON файлу для восстановления сессии")
    parser.add_argument("--save", type=str, help="Путь к JSON файлу для сохранения сессии")
    parser.add_argument("--multi-agent", action="store_true", help="Запустить в режиме Multi-Agent (Researcher + Reviewer)")
    parser.add_argument("--planner", action="store_true", help="Запустить в режиме Planner -> Multi-Agent")

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

    if args.planner:
        run_planner_agent_loop(args.goal, client, model_name=args.model, max_steps=args.max_steps)
    elif args.multi_agent:
        run_multi_agent_loop(args.goal, client, model_name=args.model, max_steps=args.max_steps)
    else:
        run_agent_loop(args.goal, client, model_name=args.model, max_steps=args.max_steps, resume_path=args.resume, save_path=args.save)
