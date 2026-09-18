The user wants to formulate a plan, execute the next step, update `next_step.md`, and adjust `TODO.md` if necessary.
Looking at `TODO.md`, all tasks are checked off.
The most recent task was: "Добавить функцию compute_cohens_d_between_models в src/experiment_utils.py и написать тесты."
And earlier: "Создать файл docs/api_reference.md для описания всех добавленных метрик и утилит из src/experiment_utils.py."

However, I ran a script that checked `docs/api_reference.md` and found that several recently added functions are missing from the documentation:
`compute_paired_t_statistic_between_models`, `compute_wilcoxon_signed_rank_statistic_between_models`, `compute_kruskal_wallis_statistic_between_models`, `compute_anova_statistic_between_models`, `compute_friedman_statistic_between_models`, `compute_cohens_d_between_models`.

Also, I checked `src/experiment_utils.py` and there are missing tasks for other effect sizes. Let me check `TODO.md` again. Wait, I should add `compute_hedges_g_between_models` as the next step, since Cohen's d was just added and Hedges' g is a natural follow-up for effect sizes. Or maybe I should update the documentation for the missing functions.

Let's look at `next_step.md`.
