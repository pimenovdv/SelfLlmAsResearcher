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

if __name__ == '__main__':
    unittest.main()
