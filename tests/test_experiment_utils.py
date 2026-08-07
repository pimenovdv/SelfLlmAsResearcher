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

if __name__ == '__main__':
    unittest.main()
