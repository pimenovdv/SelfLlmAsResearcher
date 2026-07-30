# Следующий шаг (Next Step)

**Задача:** Масштабирование экспериментов In-Context Learning на другие архитектуры (GPT-Neo и LLaMA).

**Описание:**
Мы успешно задокументировали механизмы (Circuit) для задачи In-Context Learning в модели gpt2, выявив роль Induction Heads.
Следующий логичный шаг — проверить, переносится ли эта схема на другие архитектуры. Необходимо провести аналогичные эксперименты (Activation Patching, анализ Induction Heads) на моделях `EleutherAI/gpt-neo-125m` и `JackFram/llama-160m`.

**План действий на следующий этап:**
1. Адаптировать скрипты Activation Patching (слои, головы) для In-Context Learning под архитектуру GPT-Neo (`EleutherAI/gpt-neo-125m`).
2. Адаптировать скрипты для архитектуры LLaMA (`JackFram/llama-160m`).
3. Сравнить выявленные Induction Heads и пути передачи информации между тремя архитектурами (GPT-2, GPT-Neo, LLaMA).
4. Обновить `docs/icl_circuit.md` (или создать новые файлы) для фиксации результатов сравнительного анализа.
