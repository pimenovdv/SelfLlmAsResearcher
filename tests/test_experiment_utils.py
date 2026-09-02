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

    def test_get_model_device_map(self):
        import torch.nn as nn
        from src.experiment_utils import get_model_device_map

        model = nn.Linear(10, 10)
        device_map = get_model_device_map(model)
        self.assertEqual(len(device_map), 2)
        self.assertIn('weight', device_map)
        self.assertIn('bias', device_map)

    def test_has_nan_parameters(self):
        import torch
        import torch.nn as nn
        from src.experiment_utils import has_nan_parameters

        model = nn.Linear(10, 10)
        self.assertFalse(has_nan_parameters(model))

        with torch.no_grad():
            model.weight[0, 0] = float('nan')

        self.assertTrue(has_nan_parameters(model))

    def test_has_inf_parameters(self):
        import torch
        import torch.nn as nn
        from src.experiment_utils import has_inf_parameters

        model = nn.Linear(10, 10)
        self.assertFalse(has_inf_parameters(model))

        with torch.no_grad():
            model.weight[0, 0] = float('inf')

        self.assertTrue(has_inf_parameters(model))

    def test_replace_module(self):
        import torch.nn as nn
        from src.experiment_utils import replace_module

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU())

        model = DummyModel()
        new_relu = nn.GELU()

        # Replace direct child
        replace_module(model, "layer1", new_relu)
        self.assertIsInstance(model.layer1, nn.GELU)

        # Replace nested child
        replace_module(model, "layer2.1", new_relu)
        self.assertIsInstance(model.layer2[1], nn.GELU)

        with self.assertRaises(ValueError):
            replace_module(model, "nonexistent_layer", new_relu)

        with self.assertRaises(ValueError):
            replace_module(model, "layer2.5", new_relu)

    def test_get_parameter_by_name(self):
        import torch.nn as nn
        from src.experiment_utils import get_parameter_by_name

        model = nn.Linear(10, 10)
        param = get_parameter_by_name(model, "weight")
        self.assertEqual(param.shape, (10, 10))

        with self.assertRaises(ValueError):
            get_parameter_by_name(model, "nonexistent")

    def test_get_model_sparsity(self):
        import torch.nn as nn
        import torch
        from src.experiment_utils import get_model_sparsity

        model = nn.Linear(2, 2)
        self.assertEqual(get_model_sparsity(model), 0.0)

        with torch.no_grad():
            model.weight[0, 0] = 0.0

        self.assertGreater(get_model_sparsity(model), 0.0)

    def test_find_modules_by_class(self):
        import torch.nn as nn
        from src.experiment_utils import find_modules_by_class

        model = nn.Sequential(nn.Linear(10, 5), nn.Dropout(0.1), nn.Linear(5, 2))
        names = find_modules_by_class(model, nn.Linear)
        self.assertEqual(names, ['0', '2'])

        names_dropout = find_modules_by_class(model, nn.Dropout)
        self.assertEqual(names_dropout, ['1'])

    def test_check_model_weights_equality(self):
        import torch
        import torch.nn as nn
        from src.experiment_utils import check_model_weights_equality

        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        model2.load_state_dict(model1.state_dict())

        self.assertTrue(check_model_weights_equality(model1, model2))

        with torch.no_grad():
            model2.weight[0, 0] += 0.1

        self.assertFalse(check_model_weights_equality(model1, model2))

    def test_interpolate_model_weights(self):
        import torch
        import torch.nn as nn
        from src.experiment_utils import interpolate_model_weights

        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)

        with torch.no_grad():
            model1.weight.fill_(1.0)
            model1.bias.fill_(1.0)
            model2.weight.fill_(2.0)
            model2.bias.fill_(2.0)

        model_interp = interpolate_model_weights(model1, model2, alpha=0.5)

        for p_interp in model_interp.parameters():
            self.assertTrue(torch.allclose(p_interp, torch.full_like(p_interp, 1.5)))

        model_interp_0 = interpolate_model_weights(model1, model2, alpha=0.0)
        for p_interp_0 in model_interp_0.parameters():
            self.assertTrue(torch.allclose(p_interp_0, torch.full_like(p_interp_0, 1.0)))

        model_interp_1 = interpolate_model_weights(model1, model2, alpha=1.0)
        for p_interp_1 in model_interp_1.parameters():
            self.assertTrue(torch.allclose(p_interp_1, torch.full_like(p_interp_1, 2.0)))


    def test_add_noise_to_weights(self):
        import torch
        from src.experiment_utils import add_noise_to_weights
        model = torch.nn.Linear(10, 10)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)

        model_noisy = add_noise_to_weights(model, noise_std=0.1)

        # Check shapes are the same
        self.assertEqual(model.weight.shape, model_noisy.weight.shape)
        self.assertEqual(model.bias.shape, model_noisy.bias.shape)

        # Check weights are different due to noise
        self.assertFalse(torch.allclose(model.weight, model_noisy.weight))
        self.assertFalse(torch.allclose(model.bias, model_noisy.bias))

    def test_compute_cosine_similarity_between_models(self):
        import torch
        from src.experiment_utils import compute_cosine_similarity_between_models

        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(10, 5)

        with torch.no_grad():
            model1.weight.fill_(1.0)
            model1.bias.fill_(1.0)
            model2.weight.fill_(1.0)
            model2.bias.fill_(1.0)

        sim = compute_cosine_similarity_between_models(model1, model2)
        self.assertAlmostEqual(sim, 1.0, places=5)

        with torch.no_grad():
            model2.weight.fill_(-1.0)
            model2.bias.fill_(-1.0)

        sim = compute_cosine_similarity_between_models(model1, model2)
        self.assertAlmostEqual(sim, -1.0, places=5)

        class EmptyModel(torch.nn.Module):
            pass

        empty1 = EmptyModel()
        empty2 = EmptyModel()
        self.assertEqual(compute_cosine_similarity_between_models(empty1, empty2), 0.0)

    def test_compute_l2_distance_between_models(self):
        import torch
        from src.experiment_utils import compute_l2_distance_between_models

        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(10, 5)

        with torch.no_grad():
            model1.weight.fill_(1.0)
            model1.bias.fill_(1.0)
            model2.weight.fill_(1.0)
            model2.bias.fill_(1.0)

        dist = compute_l2_distance_between_models(model1, model2)
        self.assertAlmostEqual(dist, 0.0, places=4)

        with torch.no_grad():
            model2.weight.fill_(2.0)
            model2.bias.fill_(2.0)

        # 55 parameters: sqrt(55 * (2.0 - 1.0)^2) = sqrt(55) ~ 7.416198
        dist = compute_l2_distance_between_models(model1, model2)
        self.assertAlmostEqual(dist, 55 ** 0.5, places=4)

        class EmptyModel(torch.nn.Module):
            pass

        empty1 = EmptyModel()
        empty2 = EmptyModel()
        self.assertEqual(compute_l2_distance_between_models(empty1, empty2), 0.0)

    def test_compute_l1_distance_between_models(self):
        from src.experiment_utils import compute_l1_distance_between_models
        import torch
        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(10, 5)

        with torch.no_grad():
            model1.weight.fill_(1.0)
            model1.bias.fill_(1.0)
            model2.weight.fill_(1.0)
            model2.bias.fill_(1.0)

        dist = compute_l1_distance_between_models(model1, model2)
        self.assertAlmostEqual(dist, 0.0, places=4)

        with torch.no_grad():
            model2.weight.fill_(2.0)
            model2.bias.fill_(2.0)

        # 55 parameters: 55 * |2.0 - 1.0| = 55.0
        dist = compute_l1_distance_between_models(model1, model2)
        self.assertAlmostEqual(dist, 55.0, places=4)

        class EmptyModel(torch.nn.Module):
            pass

        empty1 = EmptyModel()
        empty2 = EmptyModel()
        self.assertEqual(compute_l1_distance_between_models(empty1, empty2), 0.0)

    def test_compute_linf_distance_between_models(self):
        from src.experiment_utils import compute_linf_distance_between_models
        import torch
        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(10, 5)

        with torch.no_grad():
            model1.weight.fill_(1.0)
            model1.bias.fill_(1.0)
            model2.weight.fill_(1.0)
            model2.bias.fill_(1.0)

        dist = compute_linf_distance_between_models(model1, model2)
        self.assertAlmostEqual(dist, 0.0, places=4)

        with torch.no_grad():
            model2.weight.fill_(2.0)
            model2.bias.fill_(2.0)

        dist = compute_linf_distance_between_models(model1, model2)
        self.assertAlmostEqual(dist, 1.0, places=4)

        class EmptyModel(torch.nn.Module):
            pass

        empty1 = EmptyModel()
        empty2 = EmptyModel()
        self.assertEqual(compute_linf_distance_between_models(empty1, empty2), 0.0)

    def test_compute_parameter_norm(self):
        from src.experiment_utils import compute_parameter_norm
        import torch
        model = torch.nn.Linear(2, 2)

        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(2.0)

        # Lp norm with p=2: sqrt(4*(1.0^2) + 2*(2.0^2)) = sqrt(4 + 8) = sqrt(12) = 3.4641016151377544
        norm_l2 = compute_parameter_norm(model, 2.0)
        self.assertAlmostEqual(norm_l2, 12**0.5, places=4)

        # Lp norm with p=1: 4*|1.0| + 2*|2.0| = 4 + 4 = 8.0
        norm_l1 = compute_parameter_norm(model, 1.0)
        self.assertAlmostEqual(norm_l1, 8.0, places=4)

        class EmptyModel(torch.nn.Module):
            pass

        empty_model = EmptyModel()
        self.assertEqual(compute_parameter_norm(empty_model), 0.0)

    def test_prune_model_weights(self):
        from src.experiment_utils import prune_model_weights
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )

        with torch.no_grad():
            model[0].weight.fill_(1.0)
            model[2].weight.fill_(1.0)

        prune_model_weights(model, 0.5)

        zeros_count_0 = (model[0].weight == 0).sum().item()
        zeros_count_2 = (model[2].weight == 0).sum().item()

        self.assertEqual(zeros_count_0, 50)
        self.assertEqual(zeros_count_2, 25)

    def test_get_parameter_statistics(self):
        from src.experiment_utils import get_parameter_statistics
        import torch
        import torch.nn as nn

        model = nn.Linear(10, 2)
        with torch.no_grad():
            model.weight.fill_(2.0)
            model.bias.fill_(1.0)

        stats = get_parameter_statistics(model)
        self.assertAlmostEqual(stats["mean"], 1.9090908765792847, places=5)
        self.assertAlmostEqual(stats["min"], 1.0, places=5)
        self.assertAlmostEqual(stats["max"], 2.0, places=5)
        self.assertTrue(stats["std"] > 0)

    def test_clip_model_weights(self):
        from src.experiment_utils import clip_model_weights
        import torch
        import torch.nn as nn

        model = nn.Linear(10, 2)
        with torch.no_grad():
            model.weight.fill_(5.0)
            model.bias.fill_(-5.0)

        clip_model_weights(model, -1.0, 1.0)

        self.assertTrue(torch.all(model.weight <= 1.0))
        self.assertTrue(torch.all(model.bias >= -1.0))

    def test_scale_model_weights(self):
        from src.experiment_utils import scale_model_weights
        import torch
        import torch.nn as nn

        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)

        scale_model_weights(model, 2.5)

        self.assertTrue(torch.allclose(model.weight, torch.tensor(2.5)))
        self.assertTrue(torch.allclose(model.bias, torch.tensor(2.5)))

    def test_compute_snr(self):
        from src.experiment_utils import compute_snr
        import torch
        signal = torch.tensor([1.0, 2.0, 3.0])
        noise = torch.tensor([0.1, 0.2, 0.3])
        # signal power = (1+4+9)/3 = 14/3
        # noise power = (0.01+0.04+0.09)/3 = 0.14/3
        # SNR = 10 * log10(14/0.14) = 10 * log10(100) = 20.0
        snr = compute_snr(signal, noise)
        self.assertAlmostEqual(snr, 20.0, places=4)

    def test_compute_psnr(self):
        from src.experiment_utils import compute_psnr
        import torch
        image_true = torch.tensor([1.0, 2.0, 3.0])
        image_test = torch.tensor([1.1, 1.9, 3.1])
        # mse = (0.01 + 0.01 + 0.01) / 3 = 0.01
        # max_val = 3.0, psnr = 10 * log10(9.0 / 0.01) = 10 * log10(900) = 29.5424
        psnr = compute_psnr(image_true, image_test, max_val=3.0)
        self.assertAlmostEqual(psnr, 29.5424, places=3)

    def test_measure_inference_time(self):
        from src.experiment_utils import measure_inference_time
        import torch
        model = torch.nn.Linear(10, 2)
        input_data = torch.randn(1, 10)
        avg_time = measure_inference_time(model, input_data, num_runs=5)
        self.assertIsInstance(avg_time, float)
        self.assertGreaterEqual(avg_time, 0.0)

    def test_get_model_size_mb(self):
        from src.experiment_utils import get_model_size_mb
        import torch
        model = torch.nn.Linear(10, 2)
        size_mb = get_model_size_mb(model)
        self.assertIsInstance(size_mb, float)
        self.assertGreater(size_mb, 0.0)

    def test_get_trainable_parameters_percentage(self):
        from src.experiment_utils import get_trainable_parameters_percentage
        import torch
        model = torch.nn.Linear(10, 2)
        percentage = get_trainable_parameters_percentage(model)
        self.assertEqual(percentage, 100.0)

        for param in model.parameters():
            param.requires_grad = False
        percentage = get_trainable_parameters_percentage(model)
        self.assertEqual(percentage, 0.0)

    def test_clone_model(self):
        from src.experiment_utils import clone_model
        import torch
        model = torch.nn.Linear(10, 2)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)
        cloned_model = clone_model(model)

        self.assertIsNot(model, cloned_model)
        self.assertTrue(torch.equal(model.weight, cloned_model.weight))

        # Модифицируем оригинал и проверяем, что копия не изменилась
        with torch.no_grad():
            model.weight.fill_(2.0)
        self.assertFalse(torch.equal(model.weight, cloned_model.weight))

    def test_shift_model_weights(self):
        from src.experiment_utils import shift_model_weights
        import torch
        model = torch.nn.Linear(10, 2)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)

        shift_model_weights(model, 2.5)

        self.assertTrue(torch.allclose(model.weight, torch.tensor(3.5)))
        self.assertTrue(torch.allclose(model.bias, torch.tensor(3.5)))

    def test_randomize_model_weights(self):
        from src.experiment_utils import randomize_model_weights
        import torch
        model = torch.nn.Linear(10, 2)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)

        randomize_model_weights(model, mean=0.0, std=1.0)

        self.assertFalse(torch.allclose(model.weight, torch.tensor(1.0)))
        self.assertFalse(torch.allclose(model.bias, torch.tensor(1.0)))

    def test_average_model_weights(self):
        import torch.nn as nn
        from src.experiment_utils import average_model_weights
        model1 = nn.Linear(2, 2)
        model2 = nn.Linear(2, 2)
        import torch
        with torch.no_grad():
            model1.weight.fill_(1.0)
            model1.bias.fill_(1.0)
            model2.weight.fill_(3.0)
            model2.bias.fill_(3.0)

        avg_model = average_model_weights([model1, model2])
        self.assertTrue(torch.allclose(avg_model.weight, torch.tensor(2.0)))
        self.assertTrue(torch.allclose(avg_model.bias, torch.tensor(2.0)))

    def test_reset_model_weights(self):
        import torch.nn as nn
        from src.experiment_utils import reset_model_weights
        model = nn.Linear(2, 2)
        import torch
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)

        reset_model_weights(model)
        self.assertFalse(torch.allclose(model.weight, torch.tensor(1.0)))
        self.assertFalse(torch.allclose(model.bias, torch.tensor(1.0)))

    def test_copy_model_weights(self):
        import torch.nn as nn
        from src.experiment_utils import copy_model_weights
        model1 = nn.Linear(2, 2)
        model2 = nn.Linear(2, 2)
        import torch
        with torch.no_grad():
            model1.weight.fill_(5.0)
            model1.bias.fill_(5.0)
            model2.weight.fill_(1.0)
            model2.bias.fill_(1.0)
        copy_model_weights(model1, model2)
        self.assertTrue(torch.allclose(model2.weight, torch.tensor(5.0)))
        self.assertTrue(torch.allclose(model2.bias, torch.tensor(5.0)))

    def test_has_nan_gradients(self):
        import torch.nn as nn
        import torch
        from src.experiment_utils import has_nan_gradients
        model = nn.Linear(2, 2)
        # Without gradients
        self.assertFalse(has_nan_gradients(model))

        # With normal gradients
        loss = model(torch.tensor([1.0, 2.0])).sum()
        loss.backward()
        self.assertFalse(has_nan_gradients(model))

        # With NaN gradients
        model.weight.grad[0, 0] = float('nan')
        self.assertTrue(has_nan_gradients(model))

    def test_has_inf_gradients(self):
        import torch.nn as nn
        import torch
        from src.experiment_utils import has_inf_gradients
        model = nn.Linear(2, 2)
        # Without gradients
        self.assertFalse(has_inf_gradients(model))

        # With normal gradients
        loss = model(torch.tensor([1.0, 2.0])).sum()
        loss.backward()
        self.assertFalse(has_inf_gradients(model))

        # With Inf gradients
        model.weight.grad[0, 0] = float('inf')
        self.assertTrue(has_inf_gradients(model))

    def test_remove_all_hooks(self):
        import torch.nn as nn
        from src.experiment_utils import remove_all_hooks
        model = nn.Linear(2, 2)
        def hook_fn(*args): pass
        model.register_forward_hook(hook_fn)
        model.register_forward_pre_hook(hook_fn)
        model.register_full_backward_hook(hook_fn)

        self.assertTrue(len(model._forward_hooks) > 0)
        remove_all_hooks(model)
        self.assertEqual(len(model._forward_hooks), 0)
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model._backward_hooks), 0)

    def test_set_dropout_prob(self):
        from src.experiment_utils import set_dropout_prob
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )
        set_dropout_prob(model, 0.1)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                self.assertEqual(module.p, 0.1)

    def test_compute_parameter_sparsity(self):
        from src.experiment_utils import compute_parameter_sparsity
        import torch
        model = torch.nn.Linear(10, 2)

        # Manually set parameters to known values
        with torch.no_grad():
            model.weight.fill_(0.0) # 20 elements
            model.bias.fill_(1.0)   # 2 elements

        # Expected sparsity: 20 zeros out of 22 total elements = 20 / 22
        sparsity = compute_parameter_sparsity(model)
        self.assertAlmostEqual(sparsity, 20.0 / 22.0)

    def test_compute_gradient_sparsity(self):
        from src.experiment_utils import compute_gradient_sparsity
        import torch
        model = torch.nn.Linear(10, 2)

        # No gradients yet
        self.assertEqual(compute_gradient_sparsity(model), 0.0)

        # Set some gradients
        model.weight.grad = torch.zeros_like(model.weight)
        model.bias.grad = torch.ones_like(model.bias)

        # Total elements: 20 + 2 = 22
        # Zeros: 20
        # Expected sparsity: 20 / 22 = 0.9090909090909091
        sparsity = compute_gradient_sparsity(model)
        self.assertAlmostEqual(sparsity, 20.0 / 22.0)

    def test_get_gradient_statistics(self):
        from src.experiment_utils import get_gradient_statistics
        import torch
        model = torch.nn.Linear(10, 2)

        # Test with no gradients
        stats = get_gradient_statistics(model)
        self.assertEqual(stats, {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0})

        # Test with gradients
        model.weight.grad = torch.full_like(model.weight, 2.0)
        model.bias.grad = torch.full_like(model.bias, 4.0)

        stats = get_gradient_statistics(model)

        # We have 20 elements with 2.0 and 2 elements with 4.0.
        # Mean = (20 * 2 + 2 * 4) / 22 = 48 / 22 = 2.181818...
        self.assertAlmostEqual(stats["mean"], 48.0 / 22.0, places=6)
        self.assertEqual(stats["min"], 2.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertTrue("std" in stats)

    def test_zero_gradients(self):
        from src.experiment_utils import zero_gradients
        import torch

        model = torch.nn.Linear(10, 2)
        model.weight.grad = torch.ones_like(model.weight)

        zero_gradients(model)

        self.assertTrue(torch.all(model.weight.grad == 0))

        zero_gradients(model, set_to_none=True)
        self.assertIsNone(model.weight.grad)

    def test_add_noise_to_gradients(self):
        from src.experiment_utils import add_noise_to_gradients
        import torch

        model = torch.nn.Linear(10, 2)
        model.weight.grad = torch.zeros_like(model.weight)
        model.bias.grad = torch.zeros_like(model.bias)

        # Add noise
        add_noise_to_gradients(model, noise_std=0.5)

        # Gradients should no longer be exactly zero
        self.assertFalse(torch.all(model.weight.grad == 0))
        self.assertFalse(torch.all(model.bias.grad == 0))

    def test_clip_gradients(self):
        from src.experiment_utils import clip_gradients
        import torch

        model = torch.nn.Linear(10, 2)
        # Create dummy gradients
        model.weight.grad = torch.ones_like(model.weight) * 10.0
        model.bias.grad = torch.ones_like(model.bias) * 10.0

        # Norm of these gradients will be sqrt(20 * 10^2 + 2 * 10^2) = sqrt(2200) ≈ 46.9
        total_norm = clip_gradients(model, max_norm=1.0)
        self.assertGreater(total_norm, 10.0)

        # After clipping, the max value in gradients should be <= 1.0
        self.assertLessEqual(model.weight.grad.abs().max().item(), 1.0)

    def test_check_nan_weights(self):
        from src.experiment_utils import check_nan_weights
        import torch
        model = torch.nn.Linear(10, 2)
        self.assertFalse(check_nan_weights(model))
        with torch.no_grad():
            model.weight.data[0, 0] = float('nan')
        self.assertTrue(check_nan_weights(model))

    def test_check_inf_weights(self):
        from src.experiment_utils import check_inf_weights
        import torch
        model = torch.nn.Linear(10, 2)
        self.assertFalse(check_inf_weights(model))
        with torch.no_grad():
            model.weight.data[0, 0] = float('inf')
        self.assertTrue(check_inf_weights(model))

    def test_freeze_model_weights(self):
        from src.experiment_utils import freeze_model_weights
        import torch
        model = torch.nn.Linear(10, 2)
        freeze_model_weights(model)
        for param in model.parameters():
            self.assertFalse(param.requires_grad)

    def test_unfreeze_model_weights(self):
        from src.experiment_utils import unfreeze_model_weights
        import torch
        model = torch.nn.Linear(10, 2)
        for param in model.parameters():
            param.requires_grad = False
        unfreeze_model_weights(model)
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

    def test_compute_parameter_variance(self):
        from src.experiment_utils import compute_parameter_variance
        import torch
        model = torch.nn.Linear(10, 2)
        variance = compute_parameter_variance(model)
        self.assertIsInstance(variance, float)
        self.assertGreaterEqual(variance, 0.0)

    def test_compute_parameter_kurtosis(self):
        from src.experiment_utils import compute_parameter_kurtosis
        import torch
        model = torch.nn.Linear(10, 2)
        kurtosis = compute_parameter_kurtosis(model)
        self.assertIsInstance(kurtosis, float)

    def test_compute_parameter_skewness(self):
        from src.experiment_utils import compute_parameter_skewness
        import torch
        model = torch.nn.Linear(10, 2)
        skewness = compute_parameter_skewness(model)
        self.assertIsInstance(skewness, float)

    def test_compute_parameter_median(self):
        from src.experiment_utils import compute_parameter_median
        import torch
        model = torch.nn.Linear(10, 2)
        median = compute_parameter_median(model)
        self.assertIsInstance(median, float)

    def test_compute_parameter_quantiles(self):
        from src.experiment_utils import compute_parameter_quantiles
        import torch
        model = torch.nn.Linear(10, 2)
        quantiles = compute_parameter_quantiles(model)
        self.assertIsInstance(quantiles, list)
        self.assertEqual(len(quantiles), 3)
        self.assertTrue(all(isinstance(x, float) for x in quantiles))

    def test_compute_gradient_variance(self):
        from src.experiment_utils import compute_gradient_variance
        import torch
        model = torch.nn.Linear(10, 2)
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        variance = compute_gradient_variance(model)
        self.assertIsInstance(variance, float)

    def test_compute_gradient_kurtosis(self):
        from src.experiment_utils import compute_gradient_kurtosis
        import torch
        model = torch.nn.Linear(10, 2)
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        kurtosis = compute_gradient_kurtosis(model)
        self.assertIsInstance(kurtosis, float)

    def test_compute_gradient_skewness(self):
        from src.experiment_utils import compute_gradient_skewness
        import torch
        model = torch.nn.Linear(10, 2)
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        skewness = compute_gradient_skewness(model)
        self.assertIsInstance(skewness, float)

    def test_compute_gradient_median(self):
        from src.experiment_utils import compute_gradient_median
        import torch
        model = torch.nn.Linear(10, 2)
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        median = compute_gradient_median(model)
        self.assertIsInstance(median, float)

    def test_compute_gradient_coefficient_of_variation(self):
        from src.experiment_utils import compute_gradient_coefficient_of_variation
        import torch
        model = torch.nn.Linear(10, 2)
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        cv = compute_gradient_coefficient_of_variation(model)
        self.assertIsInstance(cv, float)

    def test_compute_gradient_quantiles(self):
        from src.experiment_utils import compute_gradient_quantiles
        import torch
        model = torch.nn.Linear(10, 2)
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        quantiles = compute_gradient_quantiles(model)
        self.assertIsInstance(quantiles, list)
        self.assertEqual(len(quantiles), 3)
        self.assertIsInstance(quantiles[0], float)

    def test_compute_parameter_coefficient_of_variation(self):
        from src.experiment_utils import compute_parameter_coefficient_of_variation
        import torch
        model = torch.nn.Linear(10, 2)
        cv = compute_parameter_coefficient_of_variation(model)
        self.assertIsInstance(cv, float)

    def test_compute_parameter_entropy(self):
        from src.experiment_utils import compute_parameter_entropy
        import torch
        model = torch.nn.Linear(10, 2)
        entropy = compute_parameter_entropy(model)
        self.assertIsInstance(entropy, float)

    def test_compute_gradient_entropy(self):
        from src.experiment_utils import compute_gradient_entropy
        import torch
        model = torch.nn.Linear(10, 2)
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        entropy = compute_gradient_entropy(model)
        self.assertIsInstance(entropy, float)

    def test_get_module_activations(self):
        from src.experiment_utils import get_module_activations
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
        model[0].weight.data.fill_(1.0)
        model[0].bias.data.fill_(0.0)
        x = torch.ones(1, 10)
        out = get_module_activations(model, '0', x)
        self.assertEqual(out.shape, (1, 5))
        self.assertTrue(torch.allclose(out, torch.full((1, 5), 10.0)))

    def test_get_module_gradients(self):
        from src.experiment_utils import get_module_gradients
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
        x = torch.ones(1, 10)
        target = torch.ones(1, 2)
        loss_fn = torch.nn.MSELoss()
        grads = get_module_gradients(model, '0', x, target, loss_fn)
        self.assertEqual(grads.shape, (1, 5))

    def test_compute_module_parameter_norms(self):
        from src.experiment_utils import compute_module_parameter_norms
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 2))
        with torch.no_grad():
            model[0].weight.fill_(1.0)
            model[0].bias.fill_(1.0)
            model[1].weight.fill_(1.0)
            model[1].bias.fill_(1.0)
        norms = compute_module_parameter_norms(model)
        self.assertIn('0', norms)
        self.assertIn('1', norms)
        self.assertAlmostEqual(norms['0'], 55.0 ** 0.5, places=4)
        self.assertAlmostEqual(norms['1'], 12.0 ** 0.5, places=4)

    def test_compute_module_gradient_norms(self):
        from src.experiment_utils import compute_module_gradient_norms
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 2))
        x = torch.ones(1, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        norms = compute_module_gradient_norms(model)
        self.assertIn('0', norms)
        self.assertIn('1', norms)

    def test_compute_activation_norms(self):
        import torch
        from src.experiment_utils import compute_activation_norms

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)

            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        model = DummyModel()
        input_data = torch.randn(2, 10)

        norms = compute_activation_norms(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', norms)
        self.assertIn('layer2', norms)
        self.assertTrue(isinstance(norms['layer1'], float))
        self.assertTrue(isinstance(norms['layer2'], float))

    def test_compute_activation_sparsity(self):
        import torch
        from src.experiment_utils import compute_activation_sparsity

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        sparsity = compute_activation_sparsity(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', sparsity)
        self.assertIn('layer2', sparsity)
        self.assertTrue(isinstance(sparsity['layer1'], float))
        self.assertTrue(isinstance(sparsity['layer2'], float))

    def test_compute_activation_statistics(self):
        import torch
        from src.experiment_utils import compute_activation_statistics

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        stats = compute_activation_statistics(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', stats)
        self.assertIn('layer2', stats)
        self.assertIn('mean', stats['layer1'])
        self.assertIn('std', stats['layer1'])
        self.assertIn('min', stats['layer1'])
        self.assertIn('max', stats['layer1'])

    def test_compute_activation_variance(self):
        import torch
        from src.experiment_utils import compute_activation_variance

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        variances = compute_activation_variance(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', variances)
        self.assertIn('layer2', variances)
        self.assertIsInstance(variances['layer1'], float)

    def test_compute_activation_skewness(self):
        import torch
        from src.experiment_utils import compute_activation_skewness

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        skewness = compute_activation_skewness(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', skewness)
        self.assertIn('layer2', skewness)
        self.assertIsInstance(skewness['layer1'], float)

    def test_compute_activation_entropy(self):
        import torch
        from src.experiment_utils import compute_activation_entropy

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        entropies = compute_activation_entropy(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', entropies)
        self.assertIn('layer2', entropies)
        self.assertIsInstance(entropies['layer1'], float)

    def test_compute_activation_kurtosis(self):
        import torch
        from src.experiment_utils import compute_activation_kurtosis

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        kurtosis = compute_activation_kurtosis(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', kurtosis)
        self.assertIn('layer2', kurtosis)
        self.assertIsInstance(kurtosis['layer1'], float)

    def test_compute_activation_median(self):
        import torch
        from src.experiment_utils import compute_activation_median

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        medians = compute_activation_median(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', medians)
        self.assertIn('layer2', medians)
        self.assertIsInstance(medians['layer1'], float)

    def test_compute_activation_quantiles(self):
        import torch
        from src.experiment_utils import compute_activation_quantiles

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        quantiles = compute_activation_quantiles(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', quantiles)
        self.assertIn('layer2', quantiles)
        self.assertIsInstance(quantiles['layer1'], list)
        self.assertEqual(len(quantiles['layer1']), 3)

    def test_compute_activation_coefficient_of_variation(self):
        from src.experiment_utils import compute_activation_coefficient_of_variation
        import torch
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        cv_dict = compute_activation_coefficient_of_variation(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', cv_dict)
        self.assertIn('layer2', cv_dict)
        self.assertIsInstance(cv_dict['layer1'], float)

    def test_compute_activation_range(self):
        from src.experiment_utils import compute_activation_range
        import torch
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        model = DummyModel()
        input_data = torch.randn(2, 10)
        range_dict = compute_activation_range(model, input_data, ['layer1', 'layer2'])
        self.assertIn('layer1', range_dict)
        self.assertIn('layer2', range_dict)
        self.assertIsInstance(range_dict['layer1'], float)
        self.assertGreaterEqual(range_dict['layer1'], 0.0)

    def test_compute_parameter_mean(self):
        from src.experiment_utils import compute_parameter_mean
        import torch
        model = torch.nn.Linear(2, 2)
        model.weight.data.fill_(2.0)
        model.bias.data.fill_(4.0)
        mean = compute_parameter_mean(model)
        # weights = [2.0, 2.0, 2.0, 2.0], biases = [4.0, 4.0]
        # sum = 8 + 8 = 16, numel = 6
        # mean = 16 / 6 = 2.6666...
        self.assertAlmostEqual(mean, 2.666666, places=4)

    def test_compute_gradient_mean(self):
        from src.experiment_utils import compute_gradient_mean
        import torch
        model = torch.nn.Linear(2, 2)
        model.weight.grad = torch.full((2, 2), 1.0)
        model.bias.grad = torch.full((2,), 4.0)
        mean = compute_gradient_mean(model)
        # sum = 4 + 8 = 12, numel = 6
        # mean = 12 / 6 = 2.0
        self.assertAlmostEqual(mean, 2.0, places=4)

    def test_compute_activation_mean(self):
        from src.experiment_utils import compute_activation_mean
        import torch
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(2, 2)
                self.layer1.weight.data.fill_(1.0)
                self.layer1.bias.data.fill_(0.0)
            def forward(self, x):
                return self.layer1(x)

        model = DummyModel()
        input_data = torch.tensor([[2.0, 3.0]])
        # act = [2+3, 2+3] = [5.0, 5.0], mean = 5.0
        mean_dict = compute_activation_mean(model, input_data, ['layer1'])
        self.assertIn('layer1', mean_dict)
        self.assertAlmostEqual(mean_dict['layer1'], 5.0, places=4)

    def test_compute_parameter_rms(self):
        from src.experiment_utils import compute_parameter_rms
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_rms(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_rms(self):
        from src.experiment_utils import compute_gradient_rms
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_rms(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_rms(self):
        from src.experiment_utils import compute_activation_rms
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        input_data = torch.randn(1, 10)
        stats = compute_activation_rms(model, input_data, ['0', '1'])
        self.assertIn('0', stats)
        self.assertIn('1', stats)
        self.assertTrue(isinstance(stats['0'], float))
        self.assertTrue(isinstance(stats['1'], float))

    def test_compute_parameter_std(self):
        from src.experiment_utils import compute_parameter_std
        import torch
        model = torch.nn.Linear(2, 2)
        model.weight.data.fill_(2.0)
        model.bias.data.fill_(4.0)
        # weights = [2.0, 2.0, 2.0, 2.0], biases = [4.0, 4.0]
        # data = [2, 2, 2, 2, 4, 4]
        # mean = 16/6 = 2.6667
        # var = (4 * (2 - 2.6667)^2 + 2 * (4 - 2.6667)^2) / 5
        # var = (4 * 0.4444 + 2 * 1.7777) / 5 = (1.7777 + 3.5555) / 5 = 5.3333 / 5 = 1.0666
        # std = sqrt(1.0666) = 1.0328
        std = compute_parameter_std(model)
        self.assertAlmostEqual(std, 1.03279, places=4)

    def test_compute_gradient_std(self):
        from src.experiment_utils import compute_gradient_std
        import torch
        model = torch.nn.Linear(2, 2)
        model.weight.grad = torch.full((2, 2), 1.0)
        model.bias.grad = torch.full((2,), 4.0)
        # grads = [1, 1, 1, 1, 4, 4]
        # mean = 12/6 = 2
        # var = (4 * (1-2)^2 + 2 * (4-2)^2) / 5 = (4*1 + 2*4) / 5 = (4+8)/5 = 12/5 = 2.4
        # std = sqrt(2.4) = 1.5492
        std = compute_gradient_std(model)
        self.assertAlmostEqual(std, 1.54919, places=4)

    def test_compute_activation_std(self):
        from src.experiment_utils import compute_activation_std
        import torch
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(2, 2)
                self.layer1.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
                self.layer1.bias.data.fill_(0.0)
            def forward(self, x):
                return self.layer1(x)

        model = DummyModel()
        input_data = torch.tensor([[2.0, 4.0]])
        # act = [2.0, 4.0]
        # std = sqrt( ( (2-3)^2 + (4-3)^2 ) / 1 ) = sqrt(2) = 1.4142
        std_dict = compute_activation_std(model, input_data, ['layer1'])
        self.assertIn('layer1', std_dict)
        self.assertAlmostEqual(std_dict['layer1'], 1.41421, places=4)

    def test_compute_parameter_min(self):
        from src.experiment_utils import compute_parameter_min
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_min(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_parameter_range(self):
        from src.experiment_utils import compute_parameter_range
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_range(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_parameter_max(self):
        from src.experiment_utils import compute_parameter_max
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_max(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_min(self):
        from src.experiment_utils import compute_gradient_min
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_min(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_range(self):
        from src.experiment_utils import compute_gradient_range
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_range(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_max(self):
        from src.experiment_utils import compute_gradient_max
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_max(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_min(self):
        from src.experiment_utils import compute_activation_min
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        input_data = torch.randn(1, 10)
        stats = compute_activation_min(model, input_data, ['0'])
        self.assertIn('0', stats)
        self.assertTrue(isinstance(stats['0'], float))

    def test_compute_activation_max(self):
        from src.experiment_utils import compute_activation_max
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        input_data = torch.randn(1, 10)
        stats = compute_activation_max(model, input_data, ['0'])
        self.assertIn('0', stats)
        self.assertTrue(isinstance(stats['0'], float))

    def test_compute_parameter_sum(self):
        from src.experiment_utils import compute_parameter_sum
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_sum(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_sum(self):
        from src.experiment_utils import compute_gradient_sum
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_sum(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_sum(self):
        from src.experiment_utils import compute_activation_sum
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        input_data = torch.randn(1, 10)
        stats = compute_activation_sum(model, input_data, ['0'])
        self.assertIn('0', stats)
        self.assertTrue(isinstance(stats['0'], float))

    def test_compute_parameter_iqr(self):
        from src.experiment_utils import compute_parameter_iqr
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_iqr(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_iqr(self):
        from src.experiment_utils import compute_gradient_iqr
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_iqr(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_iqr(self):
        from src.experiment_utils import compute_activation_iqr
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        input_data = torch.randn(1, 10)
        stats = compute_activation_iqr(model, input_data, ['0', '1'])
        self.assertIn('0', stats)
        self.assertIn('1', stats)
        self.assertTrue(isinstance(stats['0'], float))
        self.assertTrue(isinstance(stats['1'], float))

    def test_compute_parameter_mode(self):
        from src.experiment_utils import compute_parameter_mode
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_mode(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_mode(self):
        from src.experiment_utils import compute_gradient_mode
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_mode(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_mode(self):
        from src.experiment_utils import compute_activation_mode
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        input_data = torch.randn(1, 10)
        stats = compute_activation_mode(model, input_data, ['0', '1'])
        self.assertIn('0', stats)
        self.assertIn('1', stats)
        self.assertTrue(isinstance(stats['0'], float))
        self.assertTrue(isinstance(stats['1'], float))

    def test_compute_parameter_mad(self):
        from src.experiment_utils import compute_parameter_mad
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_mad(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_mad(self):
        from src.experiment_utils import compute_gradient_mad
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_mad(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_mad(self):
        from src.experiment_utils import compute_activation_mad
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        input_data = torch.randn(1, 10)
        stats = compute_activation_mad(model, input_data, ['0', '1'])
        self.assertIn('0', stats)
        self.assertIn('1', stats)
        self.assertTrue(isinstance(stats['0'], float))
        self.assertTrue(isinstance(stats['1'], float))

    def test_compute_parameter_energy(self):
        from src.experiment_utils import compute_parameter_energy
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_energy(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_energy(self):
        from src.experiment_utils import compute_gradient_energy
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_energy(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_energy(self):
        from src.experiment_utils import compute_activation_energy
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        input_data = torch.randn(1, 10)
        stats = compute_activation_energy(model, input_data, ['0', '1'])
        self.assertIn('0', stats)
        self.assertIn('1', stats)
        self.assertTrue(isinstance(stats['0'], float))
        self.assertTrue(isinstance(stats['1'], float))


    def test_compute_parameter_abs_mean(self):
        from src.experiment_utils import compute_parameter_abs_mean
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_abs_mean(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_abs_mean(self):
        from src.experiment_utils import compute_gradient_abs_mean
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_abs_mean(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_abs_mean(self):
        from src.experiment_utils import compute_activation_abs_mean
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        input_data = torch.randn(1, 10)
        stats = compute_activation_abs_mean(model, input_data, ['0', '1'])
        self.assertIn('0', stats)
        self.assertIn('1', stats)
        self.assertTrue(isinstance(stats['0'], float))
        self.assertTrue(isinstance(stats['1'], float))

    def test_compute_parameter_harmonic_mean(self):
        from src.experiment_utils import compute_parameter_harmonic_mean
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_harmonic_mean(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_harmonic_mean(self):
        from src.experiment_utils import compute_gradient_harmonic_mean
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_harmonic_mean(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_harmonic_mean(self):
        from src.experiment_utils import compute_activation_harmonic_mean
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        input_data = torch.randn(1, 10)
        stats = compute_activation_harmonic_mean(model, input_data, ['0', '1'])
        self.assertIn('0', stats)
        self.assertIn('1', stats)
        self.assertTrue(isinstance(stats['0'], float))
        self.assertTrue(isinstance(stats['1'], float))

    def test_compute_parameter_geometric_mean(self):
        from src.experiment_utils import compute_parameter_geometric_mean
        import torch
        model = torch.nn.Linear(10, 10)
        val = compute_parameter_geometric_mean(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_gradient_geometric_mean(self):
        from src.experiment_utils import compute_gradient_geometric_mean
        import torch
        model = torch.nn.Linear(10, 10)
        out = model(torch.randn(1, 10))
        out.sum().backward()
        val = compute_gradient_geometric_mean(model)
        self.assertTrue(isinstance(val, float))

    def test_compute_activation_geometric_mean(self):
        from src.experiment_utils import compute_activation_geometric_mean
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        input_data = torch.randn(1, 10)
        stats = compute_activation_geometric_mean(model, input_data, ['0', '1'])
        self.assertIn('0', stats)
        self.assertIn('1', stats)
        self.assertTrue(isinstance(stats['0'], float))
        self.assertTrue(isinstance(stats['1'], float))


if __name__ == '__main__':
    unittest.main()
