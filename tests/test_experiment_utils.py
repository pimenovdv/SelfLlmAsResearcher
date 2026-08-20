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

if __name__ == '__main__':
    unittest.main()
