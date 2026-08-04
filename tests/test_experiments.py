import pytest
import glob
import os

# To check if scripts parse and compile properly as a proxy for 'working'.
# Since they require models/gpu to run, we simply check syntax to ensure test coverage exists for the syntax layer.

@pytest.mark.parametrize("script_path", glob.glob("experiments/*.py"))
def test_experiment_exists_and_compiles(script_path):
    assert os.path.exists(script_path)
    with open(script_path, "r") as f:
        code = f.read()
    compile(code, script_path, "exec")
