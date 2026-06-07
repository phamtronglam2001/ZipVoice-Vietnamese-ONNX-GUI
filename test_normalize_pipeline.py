"""Minimal tests for normalize pipeline chaining (no vinorm required)."""
from __future__ import annotations

import unittest

from period_linebreak import join_soft_breaks, newline_sentence_boundary, prepare_tts_structure
from utils import (
    build_normalize_pipeline,
    normalize_full_document,
    normalize_text_pipeline,
    post_process_text,
    split_text_for_tts,
)


class NormalizePipelineTests(unittest.TestCase):
    def test_pipeline_chains_period_break_after_vieneu(self):
        raw = "-> Một. Ví dụ"
        pipe = build_normalize_pipeline(["vieneu", "period_break"])
        out = normalize_text_pipeline(raw, pipe)
        self.assertIn("\n", out)
        self.assertIn("Một.", out)
        self.assertIn("Ví dụ", out)

    def test_post_process_preserves_newlines(self):
        text = "một.\nví dụ"
        out = post_process_text(text, apply_lower=True)
        self.assertIn("\n", out)
        self.assertEqual(out, "một.\nví dụ")

    def test_full_document_then_split(self):
        raw = "một. đọc đoạn văn"
        pipe = build_normalize_pipeline(["period_break"])
        doc = normalize_full_document(raw, pipe)
        chunks = split_text_for_tts(doc, max_chars=135)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text.strip(), "một.")

    def test_newline_sentence_adds_period(self):
        raw = "Chương 1\nNội dung"
        out = newline_sentence_boundary(raw)
        self.assertIn("Chương 1.\n", out)

    def test_join_soft_breaks_merges_pdf_lines(self):
        raw = "câu bị ngắt giữa chừng\nở đây tiếp"
        out = join_soft_breaks(raw)
        self.assertNotIn("\n", out)
        self.assertIn("ở đây tiếp", out)

    def test_period_break_on_chained_output(self):
        after_prior = "-> Một. Ví dụ"
        out = prepare_tts_structure(after_prior)
        self.assertEqual(out, "-> Một.\nVí dụ")


if __name__ == "__main__":
    unittest.main()
