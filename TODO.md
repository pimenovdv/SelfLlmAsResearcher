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
- [x] Интегрировать brier_score и кросс-энтропию в эксперименты с Greater-Than Task.
- [x] Интеграция метрик в патчинг-эксперименты Greater-Than.
- [x] Интеграция top_k_accuracy и mean_reciprocal_rank в Factual Recall.

## Новые метрики
- [x] Добавить метрику top_k_accuracy в src/metrics.py и написать тесты.
- [x] Добавить метрику mean_reciprocal_rank в src/metrics.py и написать тесты.
- [x] Добавить метрику exact_match в src/metrics.py и написать тесты.
- [x] Добавить метрику total_variation_distance в src/metrics.py и написать тесты.
- [x] Добавить метрику target_probability в src/metrics.py и написать тесты.
- [x] Добавить метрику cosine_similarity в src/metrics.py и написать тесты.
- [x] Добавить метрику chebyshev_distance в src/metrics.py и написать тесты.

## Покрытие тестами
- [x] Настроить тестовое покрытие (test coverage) для скриптов в директории `experiments/`.

## Утилиты экспериментов
- [x] Вынести повторяющийся код из скриптов экспериментов в общие утилиты `src/experiment_utils.py`.
- [x] Перевести все эксперименты в `experiments/` на использование утилиты `load_model_and_tokenizer` из `src/experiment_utils.py`.
- [x] Написать новый эксперимент для проверки гипотезы о влиянии контекста на Activation Patching (context_activation_patching.py).
- [x] Добавить метрики `top_k_accuracy` и `mean_reciprocal_rank` в анализ влияния контекста (context_activation_patching.py).

## Утилиты управления памятью
- [x] Добавить функцию `clear_memory` в `src/experiment_utils.py` и написать тесты.
- [x] Добавить функцию `get_model_memory_footprint` в `src/experiment_utils.py` и написать тесты.
- [x] Создать скрипт experiments/memory_report.py для генерации отчета по использованию памяти различными моделями.

- [x] Перевести все скрипты экспериментов на прямой импорт метрик из `src.metrics` вместо `agent_workspace/templates/metrics.py` (Memory Rule).

## Анализ Внимания (Attention Pattern)
- [x] Разработать скрипт `experiments/gpt_neo_ioi_attention_pattern.py` для анализа паттернов внимания "Name Mover Heads" в GPT-Neo 125m.

## Новые метрики
- [x] Добавить метрику euclidean_distance в src/metrics.py и написать тесты.
- [x] Добавить метрику manhattan_distance в src/metrics.py и написать тесты.
- [x] Добавить метрику minkowski_distance в src/metrics.py и написать тесты.
- [x] Добавить метрику mean_squared_error в src/metrics.py и написать тесты.
- [x] Добавить метрику mean_absolute_error в src/metrics.py и написать тесты.
- [x] Добавить метрику pearson_correlation в src/metrics.py и написать тесты.
- [x] Добавить метрику huber_loss в src/metrics.py и написать тесты.
- [x] Добавить метрику log_cosh_loss в src/metrics.py и написать тесты.
- [x] Добавить метрику root_mean_squared_error в src/metrics.py и написать тесты.
- [x] Добавить метрику r2_score в src/metrics.py и написать тесты.
- [x] Добавить метрику mean_absolute_percentage_error в src/metrics.py и написать тесты.
- [x] Добавить метрику symmetric_mean_absolute_percentage_error в src/metrics.py и написать тесты.
- [x] Добавить метрику bhattacharyya_distance в src/metrics.py и написать тесты.
- [x] Добавить метрику hellinger_distance в src/metrics.py и написать тесты.
- [x] Добавить метрику jaccard_similarity в src/metrics.py и написать тесты.
- [x] Добавить метрику renyi_divergence в src/metrics.py и написать тесты.
- [x] Добавить метрику wasserstein_distance в src/metrics.py и написать тесты.
- [x] Добавить метрику chi_square_distance в src/metrics.py и написать тесты.
- [x] Добавить метрику canberra_distance в src/metrics.py и написать тесты.
- [x] Добавить метрику bray_curtis_distance в src/metrics.py и написать тесты.
- [x] Добавить функцию compute_parameter_range в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_range в src/experiment_utils.py и написать тесты.

## Утилиты (Дополнительно)
- [x] Добавить функцию find_modules_by_class в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию set_seed в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию count_parameters в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_device в src/experiment_utils.py и написать тесты.
- [x] Добавить функции freeze_model_parameters и unfreeze_model_parameters в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_module_by_name в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_model_device в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию check_model_device_consistency в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_norm в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию set_requires_grad в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_model_dtype в src/experiment_utils.py и написать тесты.
- [x] Добавить функции save_model_weights и load_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_model_device_map в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию has_nan_parameters в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию has_inf_parameters в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию replace_module в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_parameter_by_name в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_model_sparsity в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию check_model_weights_equality в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию interpolate_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию add_noise_to_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_cosine_similarity_between_models в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_l2_distance_between_models в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_l1_distance_between_models в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_linf_distance_between_models в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_norm в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию prune_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_parameter_statistics в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию clip_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию scale_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_snr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_psnr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию measure_inference_time в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_model_size_mb в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_trainable_parameters_percentage в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию clone_model в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию shift_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию randomize_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию average_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию reset_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию copy_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию has_nan_gradients в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию has_inf_gradients в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию remove_all_hooks в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию set_dropout_prob в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_sparsity в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_gradient_statistics в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию add_noise_to_gradients в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию clip_gradients в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию zero_gradients в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию check_nan_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию check_inf_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию freeze_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию unfreeze_model_weights в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_variance в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_kurtosis в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_skewness в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_median в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_quantiles в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_variance в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_kurtosis в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_skewness в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_median в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_quantiles в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_coefficient_of_variation в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_coefficient_of_variation в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_entropy в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_entropy в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_module_activations в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию get_module_gradients в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_module_parameter_norms в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_module_gradient_norms в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_norms в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_statistics в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_sparsity в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_entropy в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_variance в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_skewness в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_kurtosis в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_median в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_quantiles в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_coefficient_of_variation в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_range в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_std в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_std в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_std в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_min в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_max в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_min в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_max в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_min в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_max в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_sum в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_sum в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_sum в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_rms в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_rms в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_rms в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_iqr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_iqr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_iqr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_mad в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_mad в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_mad в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_mode в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_mode в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_mode в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_energy в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_energy в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_energy в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_abs_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_abs_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_abs_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_sparsity в src/experiment_utils.py и написать тесты.
- [x] Удалить дубликаты функций compute_parameter_abs_mean, compute_gradient_abs_mean, compute_activation_abs_mean из src/experiment_utils.py.
- [x] Добавить функцию compute_parameter_harmonic_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_harmonic_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_harmonic_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_geometric_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_geometric_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_geometric_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_gini в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_gini в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_gini в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_outlier_ratio в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_outlier_ratio в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_outlier_ratio в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_proportion_positive в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_proportion_positive в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_proportion_positive в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_proportion_negative в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_proportion_negative в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_proportion_negative в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_proportion_zero в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_proportion_zero в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_proportion_zero в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_trimmed_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_trimmed_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_trimmed_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_winsorized_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_winsorized_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_winsorized_mean в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_sem в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_sem в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_sem в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_vmr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_vmr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_vmr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_snr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_snr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_snr в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_crest_factor в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_crest_factor в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_crest_factor в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_form_factor в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_form_factor в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_form_factor в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_midrange в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_midrange в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_midrange в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_parameter_interdecile_range в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_gradient_interdecile_range в src/experiment_utils.py и написать тесты.
- [x] Добавить функцию compute_activation_interdecile_range в src/experiment_utils.py и написать тесты.
