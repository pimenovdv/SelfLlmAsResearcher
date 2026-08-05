import unittest
from unittest.mock import patch, MagicMock
from src.experiment_utils import load_model_and_tokenizer

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

if __name__ == '__main__':
    unittest.main()
