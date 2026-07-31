import unittest
from unittest.mock import patch, MagicMock
from src.mcp_tools import resolve_library_id, get_library_docs
import urllib.error

class TestMCPTools(unittest.TestCase):
    def test_resolve_library_id_known(self):
        self.assertEqual(resolve_library_id("transformer lens"), "transformer-lens")
        self.assertEqual(resolve_library_id("pytorch"), "torch")
        self.assertEqual(resolve_library_id("torch"), "torch")
        self.assertEqual(resolve_library_id("einops"), "einops")
        self.assertEqual(resolve_library_id("transformers"), "transformers")
        self.assertEqual(resolve_library_id("huggingface"), "transformers")
        self.assertEqual(resolve_library_id("accelerate"), "accelerate")
        self.assertEqual(resolve_library_id("openai"), "openai")

    def test_resolve_library_id_unknown(self):
        self.assertEqual(resolve_library_id("unknown library"), "unknown-library")
        self.assertEqual(resolve_library_id("  some package  "), "some-package")

    @patch('urllib.request.urlopen')
    def test_get_library_docs_success_no_topic(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"info": {"description": "This is a test description."}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        docs = get_library_docs("test-lib")
        self.assertIn("Documentation for test-lib:", docs)
        self.assertIn("This is a test description.", docs)

    @patch('urllib.request.urlopen')
    def test_get_library_docs_success_with_topic_found(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"info": {"description": "Paragraph 1.\\n\\nParagraph with topic testing.\\n\\nParagraph 3."}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        docs = get_library_docs("test-lib", topic="testing")
        self.assertIn("Documentation for test-lib on topic 'testing':", docs)
        self.assertIn("Paragraph with topic testing.", docs)
        self.assertNotIn("Paragraph 1.", docs)
        self.assertNotIn("Paragraph 3.", docs)

    @patch('urllib.request.urlopen')
    def test_get_library_docs_success_with_topic_not_found(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"info": {"description": "Paragraph 1.\\n\\nParagraph 2."}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        docs = get_library_docs("test-lib", topic="missing")
        self.assertIn("Topic 'missing' not found in the main documentation.", docs)
        self.assertIn("Paragraph 1.\n\nParagraph 2.", docs)

    @patch('urllib.request.urlopen')
    def test_get_library_docs_http_error(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 404
        mock_urlopen.return_value.__enter__.return_value = mock_response

        docs = get_library_docs("missing-lib")
        self.assertIn("Failed to fetch documentation for 'missing-lib'. HTTP status: 404", docs)

    @patch('urllib.request.urlopen')
    def test_get_library_docs_exception(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection error")

        docs = get_library_docs("error-lib")
        self.assertIn("Error fetching documentation for 'error-lib': Connection error", docs)

    @patch('urllib.request.urlopen')
    def test_get_library_docs_fallback_summary(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"info": {"description": "", "summary": "This is a summary fallback."}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        docs = get_library_docs("test-lib")
        self.assertIn("This is a summary fallback.", docs)

    @patch('urllib.request.urlopen')
    def test_get_library_docs_truncation(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        long_desc = "A" * 5000
        mock_response.read.return_value = f'{{"info": {{"description": "{long_desc}"}}}}'.encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        docs = get_library_docs("test-lib")
        self.assertEqual(len(docs), 4000 + len("\n...[truncated due to length]"))
        self.assertTrue(docs.endswith("\n...[truncated due to length]"))


if __name__ == '__main__':
    unittest.main()
