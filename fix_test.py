with open('tests/test_experiment_utils.py', 'r') as f:
    content = f.read()

bad_test_str = """
    def test_compute_normalized_mean_absolute_error_between_models(self):
        import torch
        from src.experiment_utils import compute_normalized_mean_absolute_error_between_models

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 10)
            def forward(self, x):
                return self.fc(x)

        model1 = DummyModel()
        model2 = DummyModel()
        stat = compute_normalized_mean_absolute_error_between_models(model1, model2)
        self.assertIsInstance(stat, float)"""
content = content.replace(bad_test_str, "")

# Insert properly before if __name__ == '__main__':
target = "if __name__ == '__main__':"
idx = content.find(target)
if idx != -1:
    new_content = content[:idx] + bad_test_str + "\n\n" + content[idx:]
    with open('tests/test_experiment_utils.py', 'w') as f:
        f.write(new_content)
