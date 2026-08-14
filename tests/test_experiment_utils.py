import unittest
from unittest.mock import patch, MagicMock
from src.experiment_utils import load_model_and_tokenizer, clear_memory

class TestExperimentUtils(unittest.TestCase):
    @patch('src.experiment_utils.AutoTokenizer.from_pretrained')
    @patch('src.experiment_utils.AutoModelForCausalLM.from_pretrained')
    def test_load_model_and_tokenizer(self, mock_model_cls, mock_tok_cls):
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = '<eos>'

        mock_model_cls.return_value = mock_model
        mock_tok_cls.return_value = mock_tok

        model, tokenizer = load_model_and_tokenizer('gpt2')

        mock_model_cls.assert_called_once_with('gpt2', output_attentions=False)
        mock_tok_cls.assert_called_once_with('gpt2')
        mock_model.eval.assert_called_once()
        self.assertEqual(tokenizer.pad_token, '<eos>')

    @patch('src.experiment_utils.torch.cuda.is_available', return_value=True)
    @patch('src.experiment_utils.torch.cuda.empty_cache')
    @patch('src.experiment_utils.gc.collect')
    def test_clear_memory_cuda_available(self, mock_gc, mock_empty_cache, mock_is_available):
        clear_memory()
        mock_gc.assert_called_once()
        mock_is_available.assert_called_once()
        mock_empty_cache.assert_called_once()

    @patch('src.experiment_utils.torch.cuda.is_available', return_value=False)
    @patch('src.experiment_utils.torch.cuda.empty_cache')
    @patch('src.experiment_utils.gc.collect')
    def test_clear_memory_cuda_not_available(self, mock_gc, mock_empty_cache, mock_is_available):
        clear_memory()
        mock_gc.assert_called_once()
        mock_is_available.assert_called_once()
        mock_empty_cache.assert_not_called()

    def test_get_model_memory_footprint(self):
        import torch.nn as nn
        from src.experiment_utils import get_model_memory_footprint
        model = nn.Linear(10, 10)
        mem = get_model_memory_footprint(model)
        self.assertGreater(mem, 0)

    @patch('src.experiment_utils.torch.cuda.is_available', return_value=True)
    @patch('src.experiment_utils.torch.cuda.manual_seed_all')
    @patch('src.experiment_utils.torch.manual_seed')
    @patch('src.experiment_utils.random.seed')
    def test_set_seed(self, mock_random_seed, mock_torch_seed, mock_manual_seed_all, mock_is_available):
        from src.experiment_utils import set_seed
        set_seed(42)
        mock_random_seed.assert_called_once_with(42)
        mock_torch_seed.assert_called_once_with(42)
        mock_is_available.assert_called_once()
        mock_manual_seed_all.assert_called_once_with(42)

    def test_count_parameters(self):
        import torch.nn as nn
        from src.experiment_utils import count_parameters
        model = nn.Linear(10, 5)
        res = count_parameters(model)
        self.assertEqual(res["total"], 55)
        self.assertEqual(res["trainable"], 55)

        for param in model.parameters():
            param.requires_grad = False
        res = count_parameters(model)
        self.assertEqual(res["total"], 55)
        self.assertEqual(res["trainable"], 0)

    @patch('src.experiment_utils.torch.cuda.is_available', return_value=True)
    def test_get_device_cuda(self, mock_cuda):
        from src.experiment_utils import get_device
        import torch
        device = get_device()
        self.assertEqual(device, torch.device("cuda"))
        mock_cuda.assert_called_once()

    @patch('src.experiment_utils.torch.cuda.is_available', return_value=False)
    @patch('src.experiment_utils.hasattr', return_value=True, create=True)
    @patch('src.experiment_utils.torch.backends.mps.is_available', return_value=True, create=True)
    def test_get_device_mps(self, mock_mps, mock_hasattr, mock_cuda):
        from src.experiment_utils import get_device
        import torch
        device = get_device()
        self.assertEqual(device, torch.device("mps"))
        mock_cuda.assert_called_once()
        mock_mps.assert_called_once()

    @patch('src.experiment_utils.torch.cuda.is_available', return_value=False)
    @patch('src.experiment_utils.hasattr', return_value=False, create=True)
    def test_get_device_cpu(self, mock_hasattr, mock_cuda):
        from src.experiment_utils import get_device
        import torch
        device = get_device()
        self.assertEqual(device, torch.device("cpu"))
        mock_cuda.assert_called_once()

    def test_freeze_model_parameters(self):
        import torch.nn as nn
        from src.experiment_utils import freeze_model_parameters
        model = nn.Linear(10, 5)
        freeze_model_parameters(model)
        for param in model.parameters():
            self.assertFalse(param.requires_grad)

    def test_unfreeze_model_parameters(self):
        import torch.nn as nn
        from src.experiment_utils import freeze_model_parameters, unfreeze_model_parameters
        model = nn.Linear(10, 5)
        freeze_model_parameters(model)
        unfreeze_model_parameters(model)
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

    def test_get_module_by_name(self):
        import torch.nn as nn
        from src.experiment_utils import get_module_by_name

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU())

        model = DummyModel()

        mod1 = get_module_by_name(model, "layer1")
        self.assertEqual(mod1, model.layer1)

        mod2 = get_module_by_name(model, "layer2.0")
        self.assertEqual(mod2, model.layer2[0])

        with self.assertRaises(ValueError):
            get_module_by_name(model, "nonexistent_layer")

    def test_get_model_device(self):
        import torch.nn as nn
        from src.experiment_utils import get_model_device
        model = nn.Linear(10, 5)
        device = get_model_device(model)
        self.assertEqual(device, model.weight.device)

        # Test with no parameters
        class EmptyModel(nn.Module):
            pass

        empty_model = EmptyModel()
        import torch
        self.assertEqual(get_model_device(empty_model), torch.device("cpu"))

    def test_check_model_device_consistency(self):
        import torch
        import torch.nn as nn
        from src.experiment_utils import check_model_device_consistency
        from unittest.mock import MagicMock

        model = nn.Linear(10, 5)
        self.assertTrue(check_model_device_consistency(model))

        mock_model = MagicMock(spec=nn.Module)
        param1 = MagicMock()
        param1.device = torch.device("cpu")
        param2 = MagicMock()
        param2.device = torch.device("cuda:0")
        mock_model.parameters.return_value = iter([param1, param2])

        self.assertFalse(check_model_device_consistency(mock_model))

    def test_compute_gradient_norm(self):
        from src.experiment_utils import compute_gradient_norm
        import torch

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 2)
            def forward(self, x):
                return self.linear(x)

        model = DummyModel()
        # Without gradients
        self.assertEqual(compute_gradient_norm(model), 0.0)

        # With gradients
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        norm = compute_gradient_norm(model)
        self.assertGreater(norm, 0.0)

    def test_set_requires_grad(self):
        import torch.nn as nn
        from src.experiment_utils import set_requires_grad
        model = nn.Linear(10, 5)
        set_requires_grad(model, False)
        for param in model.parameters():
            self.assertFalse(param.requires_grad)
        set_requires_grad(model, True)
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

    def test_get_model_dtype(self):
        import torch
        import torch.nn as nn
        from src.experiment_utils import get_model_dtype
        model = nn.Linear(10, 5)
        self.assertEqual(get_model_dtype(model), torch.float32)

        class EmptyModel(nn.Module):
            pass
        empty_model = EmptyModel()
        self.assertEqual(get_model_dtype(empty_model), torch.float32)

    def test_save_and_load_model_weights(self):
        import tempfile
        import os
        import torch
        import torch.nn as nn
        from src.experiment_utils import save_model_weights, load_model_weights
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pth")
            save_model_weights(model1, filepath)
            load_model_weights(model2, filepath)

            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_save_and_load_empty_model_weights(self):
        import tempfile
        import os
        import torch.nn as nn
        from src.experiment_utils import save_model_weights, load_model_weights
        class EmptyModel(nn.Module):
            pass

        model1 = EmptyModel()
        model2 = EmptyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model_empty.pth")
            save_model_weights(model1, filepath)
            load_model_weights(model2, filepath)

            self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
