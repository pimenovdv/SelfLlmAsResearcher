# TODO List (Дорожная карта)

## Инфраструктура
- [x] Настроить Dockerfile с поддержкой GPU и предустановленными библиотеками (`torch`, `transformers`, `einops`, `transformer_lens`).
- [x] Реализовать песочницу для выполнения кода (`execute_python`) с жестким таймаутом и лимитом памяти.
- [x] Интегрировать API ключи и настроить ReAct цикл.

## Эксперименты (Шаблоны)
- [x] Создать шаблон для Ablation Studies (отключение attention heads).
- [x] Создать шаблон для Activation Patching (подмена активаций MLP слоев).
- [x] Создать шаблон для извлечения активаций (Forward Hooks).
- [x] Разработать и задокументировать стандартные метрики для оценки экспериментов (Logit Difference, KL Divergence).
- [x] Создать шаблон для In-Context Learning (ICL).
- [x] Добавить шаблон для Direct Logit Attribution (DLA) в SandboxEnvironment.

## Агентный фреймворк
- [x] Обернуть инструменты агента в LangChain / AutoGen или самописный ReAct класс.
- [x] Добавить механизм обрезки вывода (если длина stdout > 2000 символов).
- [x] Реализовать механизм Self-Correction: автоматическая подача Traceback обратно агенту с промптом на исправление.
- [x] Добавить поддержку сохранения и восстановления состояния сессии.
- [x] Реализовать многоагентное взаимодействие (например, Researcher и Reviewer).
- [x] Создать роль агента-архитектора (Planner) для разбиения больших задач на подзадачи.
- [x] Добавить возможность Reviewer-агенту запускать дополнительные скрипты для верификации.

## Дальнейшее развитие
- [x] Провести тестирование на реальной задаче с использованием мощных моделей (`llama3` или `gpt-4o`) и оценить эффективность всей цепочки (Planner -> Researcher <-> Reviewer). (Добавлен mock-тест пайплайна).
- [x] Проанализировать логи тестирования, оценить снижение количества ошибок и повышение автономности системы.
- [x] Добавить RAG инструменты или MCP-сервер для получения актуальной документации (например, по `transformer_lens`), если агентам не хватает знаний (подключен реальный PyPI JSON API).
- [x] Масштабирование экспериментов по IOI на другие архитектуры. Протестирована модель `EleutherAI/gpt-neo-125m`, адаптирован `SandboxEnvironment` (переход на `AutoModelForCausalLM`).
- [x] Интеграция LLaMA: тестирование и адаптация маппинга слоев (`o_proj`). Проведен Activation Patching для задачи IOI на модели `JackFram/llama-160m`.
- [x] Точечный патчинг MLP (Targeted MLP Patching) для детального анализа GPT-Neo 125m и LLaMA-160m для выявления путей передачи информации в IOI Circuits.
- [x] Финализация документации и создание комплексной схемы (End-to-End Circuit Diagram) пути передачи информации в задаче IOI.

## Продвинутые исследования (Greater-Than Task)
- [x] Реализовать базовый эксперимент (Baseline) для задачи Greater-Than, проверяющий способность модели (gpt2) предсказывать год завершения, больший года начала.
- [x] Использование Activation Patching для поиска слоев/голов, ответственных за задачу Greater-Than.
- [x] Локализация специфичных голов (например, "Greater-Than Heads" или аналогичных компонентов) с помощью точечного Ablation.
- [x] Проведение расширенного Activation Patching на уровне отдельных Attention Heads для подтверждения роли "Greater-Than Heads" (например, L08H05, L07H10).
- [x] Визуализация Attention Patterns для головы L07H10 для проверки внимания на токен порога года.
- [x] Проведение точечного патчинга MLP-слоев для выявления путей передачи информации от "Greater-Than Heads" на поздние этапы.

## Продвинутые исследования (IOI Task)
- [x] Проверка работоспособности метрик на тестовой задаче (задача IOI).
- [x] Использование Activation Patching для поиска слоев/голов, ответственных за IOI (Indirect Object Identification).
- [x] Гранулярный Activation Patching на уровне отдельных attention heads для выявления "Name Mover Heads".
- [x] Оценка эффекта множественного ablation для Name Mover Heads.
- [x] Поиск S-Inhibition Heads или Previous Token Heads с использованием Activation Patching на ранних слоях.
- [x] Документирование выявленной подсети (Circuit) в задаче IOI для `gpt2`.
- [x] Точечный Activation Patching полносвязных слоев (MLP) с учетом позиций токенов для задачи IOI.
- [x] Анализ S-Inhibition Heads и точечный патчинг MLP для GPT-Neo 125m.
- [x] Анализ S-Inhibition Heads, точечный патчинг MLP и Ablation для LLaMA-160m. Документирование Circuit.
- [x] Создание сводной документации по выявленным S-Inhibition подсетям для GPT-Neo и LLaMA.

## Продвинутые исследования (Factual Recall Task)
- [x] Реализовать базовый эксперимент (Baseline) для задачи Factual Recall (извлечение фактологических знаний).
- [x] Использование Activation Patching для поиска слоев/голов, ответственных за Factual Recall.
- [x] Точечный патчинг MLP-слоев для выявления путей передачи информации.
- [x] Гранулярный Activation Patching на уровне отдельных Attention Heads для локализации Factual Recall Heads.
- [x] Проведение множественного Ablation для подтверждения роли Factual Recall Heads (L9H8, L10H0).
- [x] Анализ ранних слоев (Subject Processing) для Factual Recall, выявление роли Layer 0 и Layer 7 MLP.
- [x] Создание End-to-End схемы (End-to-End Circuit Diagram) для Factual Recall на базе gpt2, gpt-neo и llama.

## Продвинутые исследования (In-Context Learning)
- [x] Реализовать базовый эксперимент (Baseline) для задачи In-Context Learning (предсказание следующего слова по аналогии).
- [x] Использование Activation Patching (слои и головы) для локализации Induction Heads, ответственных за In-Context Learning.
- [x] Визуализация паттернов внимания (Attention Patterns) для Induction Heads.
- [x] Позиционный Activation Patching (Positional Patching) для определения позиций, из которых извлекается информация.
- [x] Документирование In-Context Learning Circuit.
- [x] Масштабирование In-Context Learning на архитектуры GPT-Neo и LLaMA.

## Тестирование
- [x] Написать тесты на переполнение контекста агента.
- [x] Проверить безопасность изолированной среды (блокировка сети, если не требуется).
- [x] Подготовить сложную тестовую задачу (например, извлечение конкретного типа знаний из MLP с использованием Activation Patching).
- [x] Разработка дополнительных сложных сценариев In-Context Learning с применением DLA.
- [x] Документирование Circuit для сложных сценариев ICL (translation/pattern).

## Утилиты и тестирование
- [x] Создать файл src/metrics.py с утилитами для расчета метрик (например, logit_difference).
- [x] Написать тесты для src/metrics.py.
- [x] Реализовать функцию kl_divergence в src/metrics.py и написать тесты.
- [x] Написать тесты для утилит класса ReActAgent (например, extract_code).
- [x] Написать тесты для SandboxEnvironment.
- [x] Написать тесты для mcp_tools.py.
- [x] Написать тесты для tools.py.
- [x] Добавить метрику entropy в src/metrics.py и написать тесты.
- [x] Добавить метрику js_divergence в src/metrics.py и написать тесты.
- [x] Добавить метрику perplexity в src/metrics.py и написать тесты.

- [x] Добавить метрику cross_entropy в src/metrics.py и написать тесты.
- [x] Добавить метрику brier_score в src/metrics.py и написать тесты.
- [x] Интеграция brier_score в существующие эксперименты.

## Дальнейшая интеграция
- [ ] Интегрировать brier_score и кросс-энтропию в эксперименты с Greater-Than Task.
