import os
import shutil
import tempfile
import unittest

from src.sandbox_env import SandboxEnvironment

class TestSandboxEnvironment(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.env = SandboxEnvironment(workspace_dir=self.test_dir)

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_resolve_path_within_workspace(self):
        # Test valid path resolution
        valid_path = "test_file.txt"
        resolved_path = self.env.resolve_path(valid_path)
        expected_path = os.path.abspath(os.path.join(self.test_dir, valid_path))
        self.assertEqual(resolved_path, expected_path)

        valid_nested_path = "nested/dir/test_file.txt"
        resolved_nested_path = self.env.resolve_path(valid_nested_path)
        expected_nested_path = os.path.abspath(os.path.join(self.test_dir, valid_nested_path))
        self.assertEqual(resolved_nested_path, expected_nested_path)

    def test_resolve_path_outside_workspace(self):
        # Test invalid path resolution (path traversal attack attempt)
        invalid_path = "../outside_file.txt"
        with self.assertRaises(ValueError) as context:
            self.env.resolve_path(invalid_path)
        self.assertTrue("Access denied" in str(context.exception))

        invalid_path_absolute = "/etc/passwd"
        with self.assertRaises(ValueError) as context:
            self.env.resolve_path(invalid_path_absolute)
        self.assertTrue("Access denied" in str(context.exception))

    def test_setup_templates(self):
        # Run template setup
        self.env.setup_templates()

        # Verify templates directory is created
        templates_dir = os.path.join(self.test_dir, "templates")
        self.assertTrue(os.path.isdir(templates_dir))

        # Check that specific templates exist
        expected_templates = [
            "activation_patching.py",
            "ablation.py",
            "forward_hooks.py",
            "metrics.py",
            "factual_recall.py"
        ]
        for template in expected_templates:
            template_path = os.path.join(templates_dir, template)
            self.assertTrue(os.path.isfile(template_path), f"Template {template} is missing")

if __name__ == '__main__':
    unittest.main()
