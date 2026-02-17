"""
Tests for the BPE tokenizer.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import BPETokenizer


class TestBPETokenizer:
    """Tests for BPETokenizer."""

    def test_base_vocab_size(self):
        """Base vocabulary should have 256 byte tokens."""
        tok = BPETokenizer()
        assert len(tok.vocab) == 256

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should return the original text."""
        tok = BPETokenizer()
        # After training, this should roundtrip correctly
        text = "Habari yako"
        encoded = tok.encode(text, add_bos=False, add_eos=False)
        decoded = tok.decode(encoded)
        assert decoded == text

    def test_special_tokens(self):
        """Special tokens should be registered after training."""
        tok = BPETokenizer()
        # After training:
        assert BPETokenizer.BOS_TOKEN == "<|bos|>"
        assert BPETokenizer.EOS_TOKEN == "<|eos|>"
        assert BPETokenizer.PAD_TOKEN == "<|pad|>"
        assert BPETokenizer.UNK_TOKEN == "<|unk|>"

    def test_train_increases_vocab(self):
        """Training should add merges and increase vocab beyond 256."""
        tok = BPETokenizer()
        text = "aababcabcd" * 100  # Repetitive text for easy merges
        tok.train(text, vocab_size=260)
        assert tok.vocab_size > 256

    def test_compression_ratio(self):
        """Compression ratio should be > 1.0 after training."""
        tok = BPETokenizer()
        text = "habari habari habari yako yako" * 100
        tok.train(text, vocab_size=300)
        ratio = tok.compression_ratio(text)
        assert ratio > 1.0

    def test_empty_string(self):
        """Encoding empty string should return only special tokens or empty."""
        tok = BPETokenizer()
        encoded = tok.encode("", add_bos=False, add_eos=False)
        assert encoded == []

    def test_save_load_roundtrip(self, tmp_path):
        """Saving and loading should preserve the tokenizer state."""
        tok = BPETokenizer()
        path = str(tmp_path / "test_tokenizer.json")
        tok.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.vocab_size == tok.vocab_size
