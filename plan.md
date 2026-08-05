1. **Create `src/experiment_utils.py` utility module:**
   - I will use `write_file` tool to write the new utility module containing the `load_model_and_tokenizer` function.

2. **Verify `src/experiment_utils.py` creation:**
   - I will use `list_files` tool with path `src/` to verify that `experiment_utils.py` has been correctly created.

3. **Update experiments to use `src/experiment_utils.py`:**
   - Modify experiment files (e.g., `experiments/factual_recall_baseline.py` and `experiments/dla_experiment.py`) via `replace_with_git_merge_diff` to import and use the new `load_model_and_tokenizer` function instead of directly loading the model and tokenizer from `transformers`. This prevents redundant code.
   - For `experiments/factual_recall_baseline.py`, update `test_model_factual_recall`.
   - For `experiments/dla_experiment.py`, update the `main` function.

4. **Verify modifications to experiments:**
   - I will use `read_file` on `experiments/factual_recall_baseline.py` and `experiments/dla_experiment.py` to confirm that the changes were applied correctly and that the new imports are present.

5. **Create test file for the utility:**
   - I will use `write_file` to create `tests/test_experiment_utils.py` with mock tests for the new module.

6. **Verify `tests/test_experiment_utils.py` creation:**
   - I will use `list_files` tool with path `tests/` to confirm that `test_experiment_utils.py` exists.

7. **Run the test suite:**
   - I will run the entire test suite using `run_in_bash_session` with the command `PYTHONPATH=. python3 -m pytest tests/` to ensure no regressions were introduced and the new test passes.

8. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done:**
   - I will use `pre_commit_instructions` and follow them to complete standard pre-commit steps.

9. **Submit changes:**
   - Using the `submit` tool, I will commit the work with a descriptive title and description onto a new branch.
