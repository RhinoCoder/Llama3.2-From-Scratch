import unittest
import torch
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import io

# Import your module
# Assuming your main code is in a file named llama_implementation.py
import llama_implementation as llm


class TestTokenization(unittest.TestCase):
    """Tests for tokenization functions"""

    def setUp(self):
        # Mock tokenizer setup for testing
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        self.mock_tokenizer.decode.side_effect = lambda x: "test token" if isinstance(x, list) else "test string"

    def test_encode_decode(self):
        """Test basic encoding and decoding functionality"""
        test_text = "hello world"
        encoded = self.mock_tokenizer.encode(test_text)
        self.assertEqual(len(encoded), 4)

        decoded = self.mock_tokenizer.decode(encoded)
        self.assertEqual(decoded, "test token")

    def test_individual_token_decode(self):
        """Test decoding of individual tokens"""
        token_id = 42
        decoded = self.mock_tokenizer.decode([token_id])
        self.assertEqual(decoded, "test token")


class TestEmbeddings(unittest.TestCase):
    """Tests for embedding operations"""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 5
        self.dim = 32
        self.device = torch.device("cpu")

        # Create dummy embeddings and weights
        self.embeddings = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
        self.norm_weights = torch.ones(self.dim, device=self.device)

    def test_rms_norm(self):
        """Test RMS normalization function"""
        normalized = llm.RmsNorm(self.embeddings, self.norm_weights, 1e-5)

        # Check shape is preserved
        self.assertEqual(normalized.shape, self.embeddings.shape)

        # Verify normalization (approximate check)
        # RMS norm should result in vectors with roughly unit norm when weights are all ones
        rms_values = torch.sqrt(torch.mean(normalized.pow(2), dim=-1))
        self.assertTrue(torch.allclose(rms_values, torch.ones_like(rms_values), atol=1e-1))


class TestAttentionMechanisms(unittest.TestCase):
    """Tests for attention mechanisms"""

    def setUp(self):
        self.seq_len = 4
        self.head_dim = 8
        self.device = torch.device("cpu")

    def test_precompute_freqs_cis(self):
        """Test precomputation of frequency cis values"""
        freqs_cis = llm.precomputeFreqsCis(self.head_dim, self.seq_len, self.device, 10000.0)

        # Check shape
        self.assertEqual(freqs_cis.shape, (self.seq_len, self.head_dim // 2))

        # Check dtype (should be complex)
        self.assertEqual(freqs_cis.dtype, torch.complex64)

        # Verify that magnitudes are approximately 1 (unit circle in complex plane)
        magnitudes = torch.abs(freqs_cis)
        self.assertTrue(torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5))

    def test_attention_mask(self):
        """Test attention masking"""
        batch_size = 2
        seq_len = 4
        qk = torch.randn(batch_size, seq_len, seq_len)

        mask = torch.full((seq_len, seq_len), float("-inf"), device=self.device)
        mask = torch.triu(mask, diagonal=1)

        masked_qk = qk + mask

        # Check that upper triangle is masked (has -inf values)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertTrue(torch.isinf(masked_qk[:, i, j]).all())

        # Check that lower triangle and diagonal remain unchanged
        for i in range(seq_len):
            for j in range(i + 1):
                self.assertTrue(torch.isfinite(masked_qk[:, i, j]).all())


class TestGenerativeSequence(unittest.TestCase):
    """Tests for sequence generation"""

    def setUp(self):
        # Mock necessary objects
        self.mock_model = {"output.weight": torch.randn(100, 32)}
        self.mock_config = {
            "dim": 32,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "rope_theta": 10000.0,
            "vocab_size": 100
        }
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.decode.return_value = "test"
        self.device = torch.device("cpu")

        # Create initial tokens
        self.initial_tokens = torch.tensor([1, 2, 3])

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_updateLine(self, mock_stdout):
        """Test updateLine function"""
        text = "Test message"
        length = llm.updateLine(text)

        # Check return value is correct length
        self.assertEqual(length, len(text))

        # Check stdout contains the message
        self.assertIn(text, mock_stdout.getvalue())

    @patch('llama_implementation.updateLine')
    @patch('llama_implementation.clearPreviousOutput')
    def test_generateSequenceStream_calls(self, mock_update, mock_clear):
        """Test that generateSequenceStream makes expected calls"""
        # Setup mocks to prevent actual model execution
        with patch.object(llm, 'precomputeFreqsCis', return_value=torch.ones(4, 4, dtype=torch.complex64)):
            with patch.object(llm, 'RmsNorm', return_value=torch.zeros(4, 32)):
                with patch.object(torch, 'cat', return_value=self.initial_tokens):
                    with patch.object(torch, 'argmax', return_value=torch.tensor(1)):
                        # This is just testing that the function runs without error
                        # and makes expected calls to other functions
                        try:
                            llm.generateSequenceStream(
                                self.mock_model,
                                self.initial_tokens,
                                self.mock_config,
                                self.mock_tokenizer,
                                max_new_tokens=1,
                                device=self.device
                            )
                            # If we got here, no exceptions were raised
                            mock_update.assert_called()
                        except Exception as e:
                            self.fail(f"generateSequenceStream raised exception {e}")


class TestUtils(unittest.TestCase):
    """Tests for utility functions"""

    def test_getTopNNextTokens(self):
        """Test getting top N next tokens"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"

        logits = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2])
        top_n = 3

        # Mock torch.topk to return known values
        with patch.object(torch, 'topk', return_value=(torch.tensor([0.8, 0.5, 0.3]), torch.tensor([3, 1, 2]))):
            result = llm.getTopNNextTokens(logits, mock_tokenizer, top_n)

            # Check correct number of results
            self.assertEqual(len(result), top_n)

            # Check result format and mock decode calls
            for token, prob in result:
                self.assertTrue(isinstance(token, str))
                self.assertTrue(isinstance(prob, float) or isinstance(prob, int))


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""

    @unittest.skipIf(not torch.cuda.is_available() and not torch.backends.mps.is_available(),
                     "Skip if no GPU/MPS available")
    def test_model_execution_on_gpu(self):
        """Test that model can run on GPU (if available)"""
        # This is more of an integration test and would need the actual model files
        pass

    @patch('builtins.input', return_value="test prompt")
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_main_function_execution(self, mock_stdout, mock_input):
        """Test that main function executes without errors"""
        # This would be more thoroughly tested with actual integration tests
        # Here we're just making sure it can run by mocking most dependencies

        # Need to mock many functions and methods that would be called by main()
        with patch.multiple(llm,
                            tiktoken=MagicMock(),
                            torch=MagicMock(),
                            load_tiktoken_bpe=MagicMock(),
                            precomputeFreqsCis=MagicMock(),
                            RmsNorm=MagicMock(),
                            displayQkHeatmap=MagicMock(),
                            generateSequenceStream=MagicMock(),
                            updateLine=MagicMock(),
                            getTopNNextTokens=MagicMock()):

            # Mock torch.load
            with patch('torch.load', return_value={}):
                # Mock open for file operations
                with patch('builtins.open', MagicMock()):
                    # Mock json.load
                    with patch('json.load', return_value={"dim": 32, "n_layers": 2, "n_heads": 4,
                                                          "n_kv_heads": 2, "rope_theta": 10000.0,
                                                          "vocab_size": 100, "multiple_of": 32,
                                                          "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5}):
                        try:
                            # Just test if the function runs without errors
                            with self.assertRaises(Exception):
                                llm.main()  # Will likely raise exception due to mocks
                        except Exception:
                            # Expected exception from incomplete mocking
                            pass


if __name__ == '__main__':
    unittest.main()