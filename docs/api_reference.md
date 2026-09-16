# API Reference

## src/experiment_utils.py

### `add_noise_to_gradients(model: torch.nn.modules.module.Module, noise_std: float = 0.01) -> None`
Добавляет гауссовский шум с нулевым средним и заданным стандартным отклонением
к градиентам параметров модели (если они существуют).

Args:
    model: Модель PyTorch.
    noise_std: Стандартное отклонение шума.

---

### `add_noise_to_weights(model: torch.nn.modules.module.Module, noise_std: float = 0.01) -> torch.nn.modules.module.Module`
Создает копию модели и добавляет гауссовский шум с заданным стандартным отклонением к ее весам.

---

### `average_model_weights(models: list[torch.nn.modules.module.Module]) -> torch.nn.modules.module.Module`
Усредняет веса списка моделей с одинаковой архитектурой.

---

### `check_inf_weights(model: torch.nn.modules.module.Module) -> bool`
Проверяет, есть ли Inf (бесконечность) значения в весах модели.

---

### `check_model_device_consistency(model: torch.nn.modules.module.Module) -> bool`
Проверяет, находятся ли все параметры модели на одном устройстве.

---

### `check_model_weights_equality(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> bool`
Проверяет, равны ли веса двух моделей.

---

### `check_nan_weights(model: torch.nn.modules.module.Module) -> bool`
Проверяет, есть ли NaN значения в весах модели.

---

### `clear_memory()`
Очищает кэш GPU и вызывает сборщик мусора.

---

### `clip_gradients(model: torch.nn.modules.module.Module, max_norm: float, norm_type: float = 2.0) -> float`
Обрезает градиенты модели (gradient clipping) по норме.

Args:
    model: Модель PyTorch.
    max_norm: Максимальная норма градиентов.
    norm_type: Тип используемой нормы (по умолчанию L2).

Returns:
    Общая норма градиентов до обрезки.

---

### `clip_model_weights(model: torch.nn.modules.module.Module, min_val: float, max_val: float) -> None`
Clips all parameters of the model to be within the range [min_val, max_val].

---

### `clone_model(model: torch.nn.modules.module.Module) -> torch.nn.modules.module.Module`
Создает и возвращает глубокую копию модели.

---

### `compute_activation_abs_mean(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет среднее абсолютное значение активаций для заданных слоев.

---

### `compute_activation_bimodality_coefficient(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет коэффициент бимодальности (Bimodality Coefficient) для активаций заданных слоев.

---

### `compute_activation_bowley_skewness(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Bowley skewness активаций для заданных слоев модели.

---

### `compute_activation_coefficient_of_range(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет коэффициент размаха (Coefficient of Range) активаций для заданных слоев ((Max - Min) / (Max + Min)).

---

### `compute_activation_coefficient_of_variation(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет коэффициент вариации (Coefficient of Variation, std / mean) активаций для заданных слоев при проходе input_data.

---

### `compute_activation_crest_factor(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Crest Factor (Peak-to-Average Ratio) активаций для заданных слоев.

---

### `compute_activation_crows_siddiqui_kurtosis(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Crow-Siddiqui Kurtosis для активаций заданных слоев.

---

### `compute_activation_energy(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет энергию (сумму квадратов значений) активаций для заданных слоев модели.

---

### `compute_activation_entropy(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str], bins: int = 256) -> dict[str, float]`
Вычисляет энтропию активаций для заданных слоев, оценивая распределение через гистограмму.

---

### `compute_activation_form_factor(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Form Factor (коэффициент формы) активаций для заданных слоев (RMS / Mean Abs).

---

### `compute_activation_gearys_kurtosis(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`

---

### `compute_activation_geometric_mean(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет среднее геометрическое активаций для заданных слоев (по абсолютным значениям).

---

### `compute_activation_gini(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет коэффициент Джини активаций для заданных слоев.

---

### `compute_activation_harmonic_mean(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет среднее гармоническое активаций для заданных слоев (по абсолютным значениям).

---

### `compute_activation_hoyer_sparsity(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет разреженность Хойера (Hoyer's Sparsity) активаций для заданных слоев.

---

### `compute_activation_interdecile_range(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет интердецильный размах (IDR) активаций для заданных слоев при проходе input_data.

---

### `compute_activation_interquartile_range(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет интерквартильный размах (IQR) активаций для заданных слоев (75-й процентиль минус 25-й процентиль).

---

### `compute_activation_iqr(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет межквартильный размах (IQR) активаций для заданных слоев при проходе input_data.

---

### `compute_activation_jarque_bera(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет статистику критерия Харке-Бера (Jarque-Bera) для активаций заданных слоев.

---

### `compute_activation_kelly_skewness(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Kelly skewness активаций для заданных слоев модели.

---

### `compute_activation_kurtosis(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет эксцесс (kurtosis) активаций для заданных слоев.

---

### `compute_activation_mad(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет среднее абсолютное отклонение (MAD) активаций для заданных слоев при проходе input_data.

---

### `compute_activation_max(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет максимальное значение активаций для каждого указанного слоя.

---

### `compute_activation_mean(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет среднее значение активаций для заданных слоев модели.

---

### `compute_activation_mean_absolute_deviation(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет среднее абсолютное отклонение (Mean Absolute Deviation) активаций для заданных слоев при проходе input_data.

---

### `compute_activation_median(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет медиану активаций для заданных слоев.

---

### `compute_activation_midhinge(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет midhinge активаций для заданных слоев модели ((Q1 + Q3) / 2).

---

### `compute_activation_midrange(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет полуразмах (midrange) активаций для заданных слоев: (max + min) / 2.

---

### `compute_activation_min(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет минимальное значение активаций для каждого указанного слоя.

---

### `compute_activation_mode(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет моду (наиболее частое значение) активаций для заданных слоев модели.

---

### `compute_activation_moors_kurtosis(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Moors Kurtosis для активаций заданных слоев.

---

### `compute_activation_norms(model: torch.nn.modules.module.Module, input_data: torch.Tensor, module_names: list, p: float = 2.0) -> dict`
Вычисляет Lp норму активаций для списка модулей при прохождении input_data через модель.
Возвращает словарь {module_name: norm}.

---

### `compute_activation_outlier_ratio(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str], threshold: float = 3.0) -> dict[str, float]`
Вычисляет долю выбросов среди активаций для заданных слоев.

---

### `compute_activation_pearsons_median_skewness(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Pearson's median skewness активаций для заданных слоев модели.

---

### `compute_activation_pearsons_mode_skewness(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Pearson's mode skewness активаций заданных слоев.

---

### `compute_activation_proportion_negative(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет долю отрицательных элементов активаций для заданных слоев.

---

### `compute_activation_proportion_positive(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет долю положительных элементов активаций для заданных слоев.

---

### `compute_activation_proportion_zero(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет долю нулевых элементов активаций для заданных слоев.

---

### `compute_activation_quantiles(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str], q: list[float] = None) -> dict[str, list[float]]`
Вычисляет квантили активаций для заданных слоев при проходе input_data.

---

### `compute_activation_quartile_coefficient_of_dispersion(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет квартильный коэффициент дисперсии (Quartile Coefficient of Dispersion) активаций для заданных слоев ((Q3 - Q1) / (Q3 + Q1)).

---

### `compute_activation_range(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет размах (range = max - min) активаций для заданных слоев при проходе input_data.

---

### `compute_activation_renyi_entropy(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str], alpha: float = 2.0, bins: int = 256) -> dict[str, float]`
Вычисляет энтропию Реньи активаций для заданных слоев.

---

### `compute_activation_rms(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет среднеквадратичное значение (RMS) активаций для каждого указанного слоя.

---

### `compute_activation_robust_coefficient_of_variation(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет робастный коэффициент вариации активаций для заданных слоев (MAD / Median).

---

### `compute_activation_sem(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет стандартную ошибку среднего (SEM) активаций для заданных слоев.

---

### `compute_activation_skewness(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет асимметрию (skewness) активаций для заданных слоев.

---

### `compute_activation_snr(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Signal-to-Noise Ratio (SNR) активаций для заданных слоев.

---

### `compute_activation_sparsity(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str], threshold: float = 1e-07) -> dict[str, float]`
Вычисляет разреженность активаций для заданных слоев (доля элементов, абсолютное значение которых меньше threshold).

---

### `compute_activation_statistics(model: torch.nn.modules.module.Module, input_data: torch.Tensor, module_names: list) -> dict`
Computes activation statistics (mean, std, min, max) for given layers.

---

### `compute_activation_std(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет стандартное отклонение активаций для заданных слоев модели.

---

### `compute_activation_sum(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет сумму активаций для каждого указанного слоя.

---

### `compute_activation_total_variation(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет total variation (полную вариацию) активаций для заданных слоев.

---

### `compute_activation_trimean(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет trimean активаций для заданных слоев.

---

### `compute_activation_trimmed_mean(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str], trim_percent: float = 0.1) -> dict[str, float]`
Вычисляет усеченное среднее (trimmed mean) активаций для заданных слоев.

---

### `compute_activation_tsallis_entropy(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str], alpha: float = 2.0, bins: int = 256) -> dict[str, float]`
Вычисляет энтропию Тсаллиса активаций для заданных слоев.

---

### `compute_activation_variance(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет дисперсию активаций для заданных слоев.

---

### `compute_activation_vmr(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]`
Вычисляет Variance-to-Mean Ratio (VMR) активаций для заданных слоев.

---

### `compute_activation_winsorized_mean(model: torch.nn.modules.module.Module, input_data: torch.Tensor, layer_names: list[str], limits: tuple[float, float] = (0.05, 0.05)) -> dict[str, float]`
Вычисляет винзоризованное среднее (winsorized mean) активаций для заданных слоев.

---

### `compute_angular_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет угловое расстояние (Angular Distance) между весами двух моделей.

---

### `compute_bhattacharyya_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет расстояние Бхаттачарья между весами двух моделей,
рассматривая их как вероятностные распределения.

---

### `compute_bray_curtis_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет расстояние Брея-Кертиса (Bray-Curtis Distance) между распределениями абсолютных значений весов двух моделей.

---

### `compute_canberra_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет расстояние Канберры (Canberra Distance) между распределениями абсолютных значений весов двух моделей.

---

### `compute_chebyshev_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет расстояние Чебышева (максимальное отклонение) между весами двух моделей.

---

### `compute_chi_square_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет расстояние Хи-квадрат (Chi-Square Distance) между распределениями абсолютных значений весов двух моделей.

---

### `compute_concordance_correlation_coefficient_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Computes the Concordance Correlation Coefficient (CCC) between the parameters of two models.

Args:
    model1 (nn.Module): The first PyTorch model.
    model2 (nn.Module): The second PyTorch model.

Returns:
    float: The CCC between the models. Returns 0.0 if any model has no parameters or standard deviation is 0.

---

### `compute_cosine_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет косинусное расстояние (Cosine Distance) между весами двух моделей.
Равно 1.0 - cosine_similarity.

---

### `compute_cosine_similarity_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет косинусное сходство между весами двух моделей.

---

### `compute_fractional_bias_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет дробное смещение (Fractional Bias) между весами двух моделей.
FB = 2 * (mean(A) - mean(B)) / (mean(A) + mean(B))

---

### `compute_gradient_abs_mean(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее абсолютное значение градиентов модели.

---

### `compute_gradient_bimodality_coefficient(model: torch.nn.modules.module.Module) -> float`
Вычисляет коэффициент бимодальности (Bimodality Coefficient) для градиентов модели.

---

### `compute_gradient_bowley_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет Bowley skewness градиентов параметров модели.

---

### `compute_gradient_coefficient_of_range(model: torch.nn.modules.module.Module) -> float`
Вычисляет коэффициент размаха (Coefficient of Range) градиентов модели ((Max - Min) / (Max + Min)).

---

### `compute_gradient_coefficient_of_variation(model: torch.nn.modules.module.Module) -> float`
Вычисляет коэффициент вариации (coefficient of variation) градиентов модели.

---

### `compute_gradient_crest_factor(model: torch.nn.modules.module.Module) -> float`
Вычисляет Crest Factor (Peak-to-Average Ratio) всех градиентов модели.

---

### `compute_gradient_crows_siddiqui_kurtosis(model: torch.nn.modules.module.Module) -> float`
Вычисляет Crow-Siddiqui Kurtosis для градиентов модели.

---

### `compute_gradient_energy(model: torch.nn.modules.module.Module) -> float`
Вычисляет энергию (сумму квадратов значений) градиентов параметров модели.

---

### `compute_gradient_entropy(model: torch.nn.modules.module.Module, bins: int = 256) -> float`
Вычисляет энтропию градиентов модели, оценивая распределение через гистограмму.

---

### `compute_gradient_form_factor(model: torch.nn.modules.module.Module) -> float`
Вычисляет Form Factor (коэффициент формы) градиентов модели (RMS / Mean Abs).

---

### `compute_gradient_gearys_kurtosis(model: torch.nn.modules.module.Module) -> float`

---

### `compute_gradient_geometric_mean(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее геометрическое градиентов модели (по абсолютным значениям).

---

### `compute_gradient_gini(model: torch.nn.modules.module.Module) -> float`
Вычисляет коэффициент Джини для всех градиентов параметров модели.

---

### `compute_gradient_harmonic_mean(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее гармоническое градиентов модели (по абсолютным значениям).

---

### `compute_gradient_hoyer_sparsity(model: torch.nn.modules.module.Module) -> float`
Вычисляет разреженность Хойера (Hoyer's Sparsity) градиентов параметров модели.

---

### `compute_gradient_interdecile_range(model: torch.nn.modules.module.Module) -> float`
Вычисляет интердецильный размах (IDR) градиентов модели.

---

### `compute_gradient_interquartile_range(model: torch.nn.modules.module.Module) -> float`
Вычисляет интерквартильный размах (IQR) градиентов модели (75-й процентиль минус 25-й процентиль).

---

### `compute_gradient_iqr(model: torch.nn.modules.module.Module) -> float`
Вычисляет межквартильный размах (IQR) градиентов модели.

---

### `compute_gradient_jarque_bera(model: torch.nn.modules.module.Module) -> float`
Вычисляет статистику критерия Харке-Бера (Jarque-Bera) для градиентов модели.

---

### `compute_gradient_kelly_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет Kelly skewness градиентов параметров модели.

---

### `compute_gradient_kurtosis(model: torch.nn.modules.module.Module) -> float`
Вычисляет эксцесс (kurtosis) градиентов модели.

---

### `compute_gradient_mad(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее абсолютное отклонение (MAD) градиентов модели.

---

### `compute_gradient_max(model: torch.nn.modules.module.Module) -> float`
Вычисляет максимальное значение градиентов параметров модели.

---

### `compute_gradient_mean(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее значение всех градиентов модели.

---

### `compute_gradient_mean_absolute_deviation(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее абсолютное отклонение (Mean Absolute Deviation) градиентов модели от их среднего.

---

### `compute_gradient_median(model: torch.nn.modules.module.Module) -> float`
Вычисляет медиану градиентов модели.

---

### `compute_gradient_midhinge(model: torch.nn.modules.module.Module) -> float`
Вычисляет midhinge градиентов модели ((Q1 + Q3) / 2).

---

### `compute_gradient_midrange(model: torch.nn.modules.module.Module) -> float`
Вычисляет полуразмах (midrange) градиентов модели: (max + min) / 2.

---

### `compute_gradient_min(model: torch.nn.modules.module.Module) -> float`
Вычисляет минимальное значение градиентов параметров модели.

---

### `compute_gradient_mode(model: torch.nn.modules.module.Module) -> float`
Вычисляет моду (наиболее частое значение) градиентов параметров модели.

---

### `compute_gradient_moors_kurtosis(model: torch.nn.modules.module.Module) -> float`
Вычисляет Moors Kurtosis (основанный на октилях) для градиентов модели.

---

### `compute_gradient_norm(model: torch.nn.modules.module.Module) -> float`
Вычисляет L2 норму градиентов всех параметров модели.

---

### `compute_gradient_outlier_ratio(model: torch.nn.modules.module.Module, threshold: float = 3.0) -> float`
Вычисляет долю выбросов среди градиентов модели.

---

### `compute_gradient_pearsons_median_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет Pearson's median skewness градиентов параметров модели.

---

### `compute_gradient_pearsons_mode_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет Pearson's mode skewness градиентов параметров модели.

---

### `compute_gradient_proportion_negative(model: torch.nn.modules.module.Module) -> float`
Вычисляет долю отрицательных элементов градиентов модели.

---

### `compute_gradient_proportion_positive(model: torch.nn.modules.module.Module) -> float`
Вычисляет долю положительных элементов градиентов модели.

---

### `compute_gradient_proportion_zero(model: torch.nn.modules.module.Module) -> float`
Вычисляет долю нулевых элементов градиентов модели.

---

### `compute_gradient_quantiles(model: torch.nn.modules.module.Module, q: list[float] = None) -> list[float]`
Вычисляет квантили градиентов модели.

---

### `compute_gradient_quartile_coefficient_of_dispersion(model: torch.nn.modules.module.Module) -> float`
Вычисляет квартильный коэффициент дисперсии (Quartile Coefficient of Dispersion) градиентов модели ((Q3 - Q1) / (Q3 + Q1)).

---

### `compute_gradient_range(model: torch.nn.modules.module.Module) -> float`
Вычисляет размах (range = max - min) градиентов модели.

---

### `compute_gradient_renyi_entropy(model: torch.nn.modules.module.Module, alpha: float = 2.0, bins: int = 256) -> float`
Вычисляет энтропию Реньи для градиентов модели.

---

### `compute_gradient_rms(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднеквадратичное значение (RMS) градиентов параметров модели.

---

### `compute_gradient_robust_coefficient_of_variation(model: torch.nn.modules.module.Module) -> float`
Вычисляет робастный коэффициент вариации градиентов модели (MAD / Median).

---

### `compute_gradient_sem(model: torch.nn.modules.module.Module) -> float`
Вычисляет стандартную ошибку среднего (SEM) всех градиентов модели.

---

### `compute_gradient_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет асимметрию (skewness) градиентов модели.

---

### `compute_gradient_snr(model: torch.nn.modules.module.Module) -> float`
Вычисляет Signal-to-Noise Ratio (SNR) градиентов модели.

---

### `compute_gradient_sparsity(model: torch.nn.modules.module.Module, threshold: float = 1e-07) -> float`
Вычисляет разреженность градиентов модели (доля элементов градиентов, абсолютное значение которых меньше threshold).

---

### `compute_gradient_std(model: torch.nn.modules.module.Module) -> float`
Вычисляет стандартное отклонение всех градиентов модели.

---

### `compute_gradient_sum(model: torch.nn.modules.module.Module) -> float`
Вычисляет сумму градиентов параметров модели.

---

### `compute_gradient_total_variation(model: torch.nn.modules.module.Module) -> float`
Вычисляет total variation (полную вариацию) градиентов модели.

---

### `compute_gradient_trimean(model: torch.nn.modules.module.Module) -> float`
Вычисляет trimean градиентов модели.

---

### `compute_gradient_trimmed_mean(model: torch.nn.modules.module.Module, trim_percent: float = 0.1) -> float`
Вычисляет усеченное среднее (trimmed mean) градиентов модели.

---

### `compute_gradient_tsallis_entropy(model: torch.nn.modules.module.Module, alpha: float = 2.0, bins: int = 256) -> float`
Вычисляет энтропию Тсаллиса для градиентов модели.

---

### `compute_gradient_variance(model: torch.nn.modules.module.Module) -> float`
Вычисляет дисперсию градиентов модели.

---

### `compute_gradient_vmr(model: torch.nn.modules.module.Module) -> float`
Вычисляет Variance-to-Mean Ratio (VMR) градиентов всех параметров модели.

---

### `compute_gradient_winsorized_mean(model: torch.nn.modules.module.Module, limits: tuple[float, float] = (0.05, 0.05)) -> float`
Вычисляет винзоризованное среднее (winsorized mean) градиентов параметров модели.

---

### `compute_hellinger_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет расстояние Хеллингера между весами двух моделей,
рассматривая их как вероятностные распределения.

---

### `compute_huber_loss_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module, delta: float = 1.0) -> float`
Вычисляет потерю Хьюбера (Huber Loss) между весами двух моделей.

---

### `compute_index_of_agreement_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Computes the Index of Agreement (d) between the weights of two models.

---

### `compute_jaccard_similarity_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет индекс Жаккара между распределениями абсолютных значений весов двух моделей.

---

### `compute_jeffreys_divergence_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет Jeffreys divergence (симметризованную KL-дивергенцию) между весами двух моделей,
преобразуя их в вероятностные распределения (через абсолютные значения и нормализацию).

---

### `compute_js_divergence_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет Jensen-Shannon divergence между весами двух моделей.

---

### `compute_kl_divergence_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет KL-дивергенцию между весами двух моделей,
преобразуя их в вероятностные распределения (через абсолютные значения и нормализацию).

---

### `compute_l0_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет L0 расстояние между весами двух моделей (количество отличающихся параметров).

---

### `compute_l1_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет L1 расстояние (Манхэттенское расстояние) между весами двух моделей.

---

### `compute_l2_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет L2 расстояние между весами двух моделей.

---

### `compute_linf_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет L-infinity (Чебышёвское) расстояние между весами двух моделей.

---

### `compute_log_cosh_error_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет ошибку Log-Cosh между весами двух моделей.

---

### `compute_mean_absolute_error_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет среднюю абсолютную ошибку (MAE) между весами двух моделей.

---

### `compute_mean_absolute_percentage_error_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет среднюю абсолютную процентную ошибку (MAPE) между весами двух моделей.

---

### `compute_mean_squared_error_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет среднеквадратичную ошибку (MSE) между весами двух моделей.

---

### `compute_mean_squared_logarithmic_error_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет среднеквадратичную логарифмическую ошибку (MSLE) между весами двух моделей.
Ограничивает значения снизу (например, нулем) перед логарифмированием для избежания NaN.

---

### `compute_minkowski_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module, p: float = 3.0) -> float`
Вычисляет расстояние Минковского между весами двух моделей.

---

### `compute_module_gradient_norms(model: torch.nn.modules.module.Module, p: float = 2.0) -> dict`
Вычисляет Lp норму градиентов каждого модуля (слоя) в модели.
Возвращает словарь {module_name: norm}.

---

### `compute_module_parameter_norms(model: torch.nn.modules.module.Module, p: float = 2.0) -> dict`
Вычисляет Lp норму параметров каждого модуля (слоя) в модели.
Возвращает словарь {module_name: norm}.

---

### `compute_normalized_root_mean_squared_error_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет нормализованную среднеквадратичную ошибку (NRMSE) между весами двух моделей.

---

### `compute_parameter_abs_mean(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее абсолютное значение параметров модели.

---

### `compute_parameter_bimodality_coefficient(model: torch.nn.modules.module.Module) -> float`
Вычисляет коэффициент бимодальности (Bimodality Coefficient) всех параметров модели.

---

### `compute_parameter_bowley_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет Bowley skewness параметров модели.
Bowley skewness = (Q3 + Q1 - 2 * Q2) / (Q3 - Q1)

---

### `compute_parameter_coefficient_of_range(model: torch.nn.modules.module.Module) -> float`
Вычисляет коэффициент размаха (Coefficient of Range) параметров модели ((Max - Min) / (Max + Min)).

---

### `compute_parameter_coefficient_of_variation(model: torch.nn.modules.module.Module) -> float`
Вычисляет коэффициент вариации (coefficient of variation) параметров модели.

---

### `compute_parameter_crest_factor(model: torch.nn.modules.module.Module) -> float`
Вычисляет Crest Factor (Peak-to-Average Ratio) всех параметров модели.

---

### `compute_parameter_crows_siddiqui_kurtosis(model: torch.nn.modules.module.Module) -> float`
Вычисляет Crow-Siddiqui Kurtosis для параметров модели.

---

### `compute_parameter_energy(model: torch.nn.modules.module.Module) -> float`
Вычисляет энергию (сумму квадратов значений) параметров модели.

---

### `compute_parameter_entropy(model: torch.nn.modules.module.Module, bins: int = 256) -> float`
Вычисляет энтропию параметров модели, оценивая распределение через гистограмму.

---

### `compute_parameter_form_factor(model: torch.nn.modules.module.Module) -> float`
Вычисляет Form Factor (коэффициент формы) параметров модели (RMS / Mean Abs).

---

### `compute_parameter_gearys_kurtosis(model: torch.nn.modules.module.Module) -> float`

---

### `compute_parameter_geometric_mean(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее геометрическое параметров модели (по абсолютным значениям).

---

### `compute_parameter_gini(model: torch.nn.modules.module.Module) -> float`
Вычисляет коэффициент Джини для всех параметров модели.

---

### `compute_parameter_harmonic_mean(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее гармоническое параметров модели (по абсолютным значениям).

---

### `compute_parameter_hoyer_sparsity(model: torch.nn.modules.module.Module) -> float`
Вычисляет разреженность Хойера (Hoyer's Sparsity) всех параметров модели.

---

### `compute_parameter_interdecile_range(model: torch.nn.modules.module.Module) -> float`
Вычисляет интердецильный размах (IDR) параметров модели (90-й процентиль минус 10-й процентиль).

---

### `compute_parameter_interquartile_range(model: torch.nn.modules.module.Module) -> float`
Вычисляет интерквартильный размах (IQR) параметров модели (75-й процентиль минус 25-й процентиль).

---

### `compute_parameter_iqr(model: torch.nn.modules.module.Module) -> float`
Вычисляет межквартильный размах (IQR) параметров модели (75-й процентиль минус 25-й процентиль).

---

### `compute_parameter_jarque_bera(model: torch.nn.modules.module.Module) -> float`
Вычисляет статистику критерия Харке-Бера (Jarque-Bera) для параметров модели.
JB = (n / 6) * (S^2 + (1/4) * (K - 3)^2)
где n - количество параметров, S - асимметрия, K - коэффициент эксцесса.

---

### `compute_parameter_kelly_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет Kelly skewness параметров модели.
Kelly skewness = (P90 + P10 - 2 * P50) / (P90 - P10)

---

### `compute_parameter_kurtosis(model: torch.nn.modules.module.Module) -> float`
Вычисляет эксцесс (kurtosis) всех параметров модели.

---

### `compute_parameter_mad(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее абсолютное отклонение (MAD) параметров модели.

---

### `compute_parameter_max(model: torch.nn.modules.module.Module) -> float`
Вычисляет максимальное значение всех параметров модели.

---

### `compute_parameter_mean(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее значение всех параметров модели.

---

### `compute_parameter_mean_absolute_deviation(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднее абсолютное отклонение (Mean Absolute Deviation) параметров модели от их среднего.

---

### `compute_parameter_median(model: torch.nn.modules.module.Module) -> float`
Вычисляет медиану всех параметров модели.

---

### `compute_parameter_midhinge(model: torch.nn.modules.module.Module) -> float`
Вычисляет midhinge параметров модели ((Q1 + Q3) / 2).

---

### `compute_parameter_midrange(model: torch.nn.modules.module.Module) -> float`
Вычисляет полуразмах (midrange) параметров модели: (max + min) / 2.

---

### `compute_parameter_min(model: torch.nn.modules.module.Module) -> float`
Вычисляет минимальное значение всех параметров модели.

---

### `compute_parameter_mode(model: torch.nn.modules.module.Module) -> float`
Вычисляет моду (наиболее частое значение) параметров модели.

---

### `compute_parameter_moors_kurtosis(model: torch.nn.modules.module.Module) -> float`
Вычисляет Moors Kurtosis (основанный на октилях) для параметров модели.

---

### `compute_parameter_norm(model: torch.nn.modules.module.Module, p: float = 2.0) -> float`
Вычисляет Lp норму всех параметров модели.

---

### `compute_parameter_outlier_ratio(model: torch.nn.modules.module.Module, threshold: float = 3.0) -> float`
Вычисляет долю выбросов (значений, отклоняющихся от среднего более чем на threshold стандартных отклонений) среди параметров модели.

---

### `compute_parameter_pearsons_median_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет Pearson's median skewness параметров модели.
Pearson's median skewness = 3 * (Mean - Median) / Standard Deviation

---

### `compute_parameter_pearsons_mode_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет Pearson's mode skewness параметров модели.
Pearson's mode skewness = (Mean - Mode) / Standard Deviation

---

### `compute_parameter_proportion_negative(model: torch.nn.modules.module.Module) -> float`
Вычисляет долю отрицательных элементов параметров модели.

---

### `compute_parameter_proportion_positive(model: torch.nn.modules.module.Module) -> float`
Вычисляет долю положительных элементов параметров модели.

---

### `compute_parameter_proportion_zero(model: torch.nn.modules.module.Module) -> float`
Вычисляет долю нулевых элементов параметров модели.

---

### `compute_parameter_quantiles(model: torch.nn.modules.module.Module, q: list[float] = None) -> list[float]`
Вычисляет квантили параметров модели.

---

### `compute_parameter_quartile_coefficient_of_dispersion(model: torch.nn.modules.module.Module) -> float`
Вычисляет квартильный коэффициент дисперсии (Quartile Coefficient of Dispersion) параметров модели ((Q3 - Q1) / (Q3 + Q1)).

---

### `compute_parameter_range(model: torch.nn.modules.module.Module) -> float`
Вычисляет размах (range = max - min) всех параметров модели.

---

### `compute_parameter_renyi_entropy(model: torch.nn.modules.module.Module, alpha: float = 2.0, bins: int = 256) -> float`
Вычисляет энтропию Реньи для параметров модели.

---

### `compute_parameter_rms(model: torch.nn.modules.module.Module) -> float`
Вычисляет среднеквадратичное значение (RMS) всех параметров модели.

---

### `compute_parameter_robust_coefficient_of_variation(model: torch.nn.modules.module.Module) -> float`
Вычисляет робастный коэффициент вариации параметров модели (MAD / Median).

---

### `compute_parameter_sem(model: torch.nn.modules.module.Module) -> float`
Вычисляет стандартную ошибку среднего (SEM) всех параметров модели.

---

### `compute_parameter_skewness(model: torch.nn.modules.module.Module) -> float`
Вычисляет асимметрию (skewness) всех параметров модели.

---

### `compute_parameter_snr(model: torch.nn.modules.module.Module) -> float`
Вычисляет Signal-to-Noise Ratio (SNR) параметров модели (mean / std).

---

### `compute_parameter_sparsity(model: torch.nn.modules.module.Module, threshold: float = 1e-07) -> float`
Вычисляет разреженность параметров модели (доля элементов параметров, абсолютное значение которых меньше threshold).

---

### `compute_parameter_std(model: torch.nn.modules.module.Module) -> float`
Вычисляет стандартное отклонение всех параметров модели.

---

### `compute_parameter_sum(model: torch.nn.modules.module.Module) -> float`
Вычисляет сумму всех параметров модели.

---

### `compute_parameter_total_variation(model: torch.nn.modules.module.Module) -> float`
Вычисляет total variation (полную вариацию) параметров модели.

---

### `compute_parameter_trimean(model: torch.nn.modules.module.Module) -> float`
Вычисляет trimean параметров модели.

---

### `compute_parameter_trimmed_mean(model: torch.nn.modules.module.Module, trim_percent: float = 0.1) -> float`
Вычисляет усеченное среднее (trimmed mean) параметров модели.

---

### `compute_parameter_tsallis_entropy(model: torch.nn.modules.module.Module, alpha: float = 2.0, bins: int = 256) -> float`
Вычисляет энтропию Тсаллиса для параметров модели.

---

### `compute_parameter_variance(model: torch.nn.modules.module.Module) -> float`
Вычисляет дисперсию всех параметров модели.

---

### `compute_parameter_vmr(model: torch.nn.modules.module.Module) -> float`
Вычисляет Variance-to-Mean Ratio (VMR) всех параметров модели.

---

### `compute_parameter_winsorized_mean(model: torch.nn.modules.module.Module, limits: tuple[float, float] = (0.05, 0.05)) -> float`
Вычисляет винзоризованное среднее (winsorized mean) параметров модели.

---

### `compute_peak_signal_to_noise_ratio_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет Peak Signal-to-Noise Ratio (PSNR) между весами двух моделей,
где model1 - сигнал, а model2 - зашумленный сигнал.

---

### `compute_pearson_correlation_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет Pearson correlation coefficient между весами двух моделей.

---

### `compute_psnr(image_true: torch.Tensor, image_test: torch.Tensor, max_val: float) -> float`
Вычисляет Peak Signal-to-Noise Ratio (PSNR).

---

### `compute_r2_score_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет коэффициент детерминации (R^2) между весами двух моделей.

---

### `compute_relative_error_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет относительную ошибку (Relative Error) между весами двух моделей.

---

### `compute_renyi_divergence_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module, alpha: float = 2.0) -> float`
Вычисляет дивергенцию Реньи (Renyi Divergence) между распределениями абсолютных значений весов двух моделей.

---

### `compute_root_mean_squared_error_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет корень из среднеквадратичной ошибки (RMSE) между весами двух моделей.

---

### `compute_signal_to_noise_ratio_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет Signal-to-Noise Ratio (SNR) между весами двух моделей,
где model1 - сигнал, а model2 - зашумленный сигнал.

---

### `compute_snr(signal: torch.Tensor, noise: torch.Tensor) -> float`
Вычисляет Signal-to-Noise Ratio (SNR).

---

### `compute_spearman_correlation_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет Spearman rank correlation coefficient между весами двух моделей.

---

### `compute_symmetric_kl_divergence_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module, epsilon: float = 1e-08) -> float`
Computes the Symmetric Kullback-Leibler (KL) Divergence between the parameters of two models.
We compute KL(P || Q) + KL(Q || P) by treating the normalized absolute weights as probability distributions.

Args:
    model1 (nn.Module): The first PyTorch model.
    model2 (nn.Module): The second PyTorch model.
    epsilon (float): A small value to avoid division by zero or log(0).

Returns:
    float: The symmetric KL divergence.

---

### `compute_cross_entropy_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет кросс-энтропию между весами двух моделей, преобразуя их в вероятностные распределения (через абсолютные значения и нормализацию).

---

### `compute_perplexity_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет перплексию между весами двух моделей на основе кросс-энтропии.

---

### `compute_symmetric_mean_absolute_percentage_error_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет симметричную среднюю абсолютную процентную ошибку (SMAPE) между весами двух моделей.

---

### `compute_total_variation_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет Total Variation Distance (TVD) между распределениями абсолютных значений весов двух моделей.

---

### `compute_tsallis_divergence_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module, alpha: float = 2.0) -> float`
Вычисляет дивергенцию Тсаллиса (Tsallis divergence) между весами двух моделей.

---

### `compute_wasserstein_distance_between_models(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module) -> float`
Вычисляет 1D Wasserstein distance (L1 distance between sorted elements) между весами двух моделей.

---

### `copy_model_weights(source_model: torch.nn.modules.module.Module, target_model: torch.nn.modules.module.Module) -> None`
Копирует веса из source_model в target_model.

---

### `count_parameters(model: torch.nn.modules.module.Module) -> dict`
Возвращает количество параметров модели (всего и обучаемых).

---

### `find_modules_by_class(model: torch.nn.modules.module.Module, module_class: type) -> list`
Возвращает список имен модулей в модели, которые являются экземплярами указанного класса.

---

### `freeze_model_parameters(model: torch.nn.modules.module.Module)`
Замораживает все параметры модели.

---

### `freeze_model_weights(model: torch.nn.modules.module.Module) -> None`
Замораживает все веса модели (устанавливает requires_grad = False).

---

### `get_device() -> torch.device`
Возвращает доступное устройство (cuda, mps или cpu).

---

### `get_gradient_statistics(model: torch.nn.modules.module.Module) -> dict`
Возвращает статистику градиентов модели (mean, std, min, max).

---

### `get_model_device(model: torch.nn.modules.module.Module) -> torch.device`
Возвращает устройство (device), на котором находятся параметры модели.

---

### `get_model_device_map(model: torch.nn.modules.module.Module) -> dict`
Возвращает словарь, сопоставляющий имена параметров модели с их устройствами.

---

### `get_model_dtype(model: torch.nn.modules.module.Module) -> torch.dtype`
Возвращает тип данных (dtype) параметров модели.

---

### `get_model_memory_footprint(model: torch.nn.modules.module.Module) -> int`
Возвращает объем памяти, занимаемый параметрами модели, в байтах.

---

### `get_model_size_mb(model: torch.nn.modules.module.Module) -> float`
Возвращает размер модели в мегабайтах (MB).

---

### `get_model_sparsity(model: torch.nn.modules.module.Module) -> float`
Вычисляет долю нулевых параметров в модели.

---

### `get_module_activations(model: torch.nn.modules.module.Module, module_name: str, input_data: torch.Tensor, **kwargs) -> torch.Tensor`
Возвращает активации (выход) указанного модуля при прохождении input_data через модель.

---

### `get_module_by_name(model: torch.nn.modules.module.Module, module_name: str) -> torch.nn.modules.module.Module`
Возвращает модуль по его имени (например, 'transformer.h.0.mlp').

---

### `get_module_gradients(model: torch.nn.modules.module.Module, module_name: str, input_data: torch.Tensor, target: torch.Tensor, loss_fn, **kwargs) -> torch.Tensor`
Возвращает градиенты по выходу указанного модуля при прохождении input_data и вычислении loss.

---

### `get_parameter_by_name(model: torch.nn.modules.module.Module, parameter_name: str) -> torch.nn.parameter.Parameter`
Возвращает параметр модели по его имени.

---

### `get_parameter_statistics(model: torch.nn.modules.module.Module) -> dict`
Возвращает статистику параметров модели (mean, std, min, max).

---

### `get_trainable_parameters_percentage(model: torch.nn.modules.module.Module) -> float`
Возвращает процент обучаемых параметров модели.

---

### `has_inf_gradients(model: torch.nn.modules.module.Module) -> bool`
Проверяет, содержат ли градиенты параметров модели Inf значения.

---

### `has_inf_parameters(model: torch.nn.modules.module.Module) -> bool`
Проверяет, содержат ли параметры модели Inf значения.

---

### `has_nan_gradients(model: torch.nn.modules.module.Module) -> bool`
Проверяет, содержат ли градиенты параметров модели NaN значения.

---

### `has_nan_parameters(model: torch.nn.modules.module.Module) -> bool`
Проверяет, содержат ли параметры модели NaN значения.

---

### `interpolate_model_weights(model1: torch.nn.modules.module.Module, model2: torch.nn.modules.module.Module, alpha: float) -> torch.nn.modules.module.Module`
Создает копию model1, веса которой интерполированы между model1 и model2.
w_new = (1 - alpha) * w1 + alpha * w2

---

### `load_model_and_tokenizer(model_name: str, output_attentions: bool = False)`
Загружает модель и токенизатор по имени.

---

### `load_model_weights(model: torch.nn.modules.module.Module, filepath: str)`
Загружает веса модели из файла.

---

### `measure_inference_time(model: torch.nn.modules.module.Module, input_data: torch.Tensor, num_runs: int = 10) -> float`
Измеряет среднее время инференса модели на заданных входных данных.

---

### `prune_model_weights(model: torch.nn.modules.module.Module, amount: float) -> None`
Применяет L1 неструктурированный прунинг (l1_unstructured) ко всем Linear слоям модели.

---

### `randomize_model_weights(model: torch.nn.modules.module.Module, mean: float = 0.0, std: float = 1.0) -> None`
Рандомизирует веса модели, используя нормальное распределение с заданными средним и стандартным отклонением.

---

### `remove_all_hooks(model: torch.nn.modules.module.Module) -> None`
Удаляет все хуки (forward, forward_pre, backward, state_dict и т.д.) из всех модулей модели.

---

### `replace_module(model: torch.nn.modules.module.Module, module_name: str, new_module: torch.nn.modules.module.Module)`
Заменяет модуль в модели по его имени (например, 'transformer.h.0.mlp') на новый модуль.

---

### `reset_model_weights(model: torch.nn.modules.module.Module) -> None`
Сбрасывает веса модели к значениям по умолчанию, используя методы инициализации каждого модуля.

---

### `save_model_weights(model: torch.nn.modules.module.Module, filepath: str)`
Сохраняет веса модели в файл.

---

### `scale_model_weights(model: torch.nn.modules.module.Module, scale_factor: float) -> None`
Умножает все параметры модели на scale_factor.

---

### `set_dropout_prob(model: torch.nn.modules.module.Module, p: float) -> None`
Устанавливает вероятность отсева (dropout probability) для всех слоев Dropout в модели.

---

### `set_requires_grad(model: torch.nn.modules.module.Module, requires_grad: bool)`
Устанавливает requires_grad для всех параметров модели.

---

### `set_seed(seed: int = 42)`
Устанавливает seed для воспроизводимости экспериментов.

---

### `shift_model_weights(model: torch.nn.modules.module.Module, shift_value: float) -> None`
Добавляет заданное значение (shift_value) ко всем весам модели.

---

### `unfreeze_model_parameters(model: torch.nn.modules.module.Module)`
Размораживает все параметры модели.

---

### `unfreeze_model_weights(model: torch.nn.modules.module.Module) -> None`
Размораживает все веса модели (устанавливает requires_grad = True).

---

### `zero_gradients(model: torch.nn.modules.module.Module, set_to_none: bool = False) -> None`
Обнуляет градиенты всех параметров модели.

Args:
    model: Модель PyTorch.
    set_to_none: Если True, устанавливает градиенты в None вместо нулей.

---

### `compute_kendall_tau_correlation_between_models(model1: torch.nn.Module, model2: torch.nn.Module) -> float`
Вычисляет Kendall rank correlation coefficient (Kendall's tau) между весами двух моделей.

---

### `compute_kuiper_statistic_between_models(model1: torch.nn.Module, model2: torch.nn.Module) -> float`
Вычисляет статистику Кипера (Kuiper statistic) между весами двух моделей.

---

### `compute_kolmogorov_smirnov_statistic_between_models(model1: torch.nn.Module, model2: torch.nn.Module) -> float`
Вычисляет статистику Колмогорова-Смирнова между весами двух моделей.

---
