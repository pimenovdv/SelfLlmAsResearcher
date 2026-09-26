1. **Add `compute_jaccard_distance_between_models` metric to `src/experiment_utils.py`**
   - Create a function `compute_jaccard_distance_between_models(model1: torch.nn.Module, model2: torch.nn.Module)` that calculates the Jaccard distance between two models (1.0 - Jaccard similarity).
   - Use string replacement in a bash python script to insert it right before `def compute_jaccard_similarity_between_models`.

2. **Add `compute_hamming_distance_between_models` metric to `src/experiment_utils.py`**
   - Create a function `compute_hamming_distance_between_models(model1: torch.nn.Module, model2: torch.nn.Module, threshold: float = 1e-5)` that calculates the normalized Hamming distance between two models' weights.
   - Use a bash python script to append it.

3. **Add corresponding tests to `tests/test_experiment_utils.py`**
   - Add `test_compute_jaccard_distance_between_models` and `test_compute_hamming_distance_between_models`.
   - Verify the insertion.

4. **Run tests**
   - Execute `PYTHONPATH=. python3 -m pytest tests/test_experiment_utils.py`.

5. **Update tracking files**
   - Add tasks to `TODO.md` as completed.
   - Update `next_step.md` with instructions for the next steps.

6. **Complete pre-commit steps**
   - Ensure proper testing, verification, review, and reflection are done.
