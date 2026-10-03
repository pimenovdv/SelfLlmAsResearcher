import re
import os

files = ["docs/api_reference.md", "next_step.md", "src/experiment_utils.py", "tests/test_experiment_utils.py"]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    def replacer(match):
        head_content = match.group(1)
        main_content = match.group(2)

        # In next_step.md, we want main's content but with our new task.
        if file == "next_step.md":
            # head has "Add another statistical metric..."
            # main has the template with status.
            return main_content.replace("1. Ожидание новых задач от пользователя.", "1. " + head_content.strip())

        # For python files, keep HEAD since HEAD contains the new code. main might be empty or missing it.
        # But wait, if main has new code, we want to keep it.
        # Let's keep both, but head first, then main.
        if not main_content.strip():
            return head_content + "\n"
        if not head_content.strip():
            return main_content + "\n"

        return head_content + "\n" + main_content + "\n"

    resolved = re.sub(r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> origin/main-3336522528379146385", replacer, content, flags=re.DOTALL)

    with open(file, "w", encoding="utf-8") as f:
        f.write(resolved)
