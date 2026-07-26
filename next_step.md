# Следующий шаг (Next Step)

**Задача:** Детальный анализ S-Inhibition Heads и MLP patching для архитектуры LLaMA.

**Описание:**
Мы успешно интегрировали архитектуру LLaMA, адаптировали маппинг слоев (использование `o_proj` вместо `c_proj`/`out_proj`) и провели первый Activation Patching эксперимент на задаче IOI (найдены потенциальные Name Mover Heads, например, Layer 9 Head 5).
Следующий шаг — углубить исследование для моделей LLaMA:
1. Выполнить поиск S-Inhibition Heads или Previous Token Heads на ранних слоях (до Layer 9).
2. Провести гранулярный Activation Patching для полносвязных слоев (MLP) с учетом позиций токенов, как мы это делали для GPT-Neo.
3. Оценить влияние множественного Ablation на найденные Name Mover Heads.

**План действий на следующий этап:**
1. Создать скрипт для поиска S-Inhibition Heads в LLaMA (`experiments/llama_s_inhibition.py`).
2. Написать скрипт для точечного патчинга MLP для LLaMA (`experiments/llama_mlp_pos_patching.py`), учитывая структуру `gate_proj`, `up_proj`, `down_proj`.
3. Задокументировать выявленную подсеть (Circuit) для LLaMA-160m.