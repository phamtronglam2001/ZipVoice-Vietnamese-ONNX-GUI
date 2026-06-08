"""Minimal tests for normalize pipeline chaining."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from period_linebreak import join_soft_breaks, newline_sentence_boundary, prepare_tts_structure
from utils import (
    build_normalize_pipeline,
    normalize_full_document,
    normalize_text_pipeline,
    post_process_text,
    read_text_file,
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
        merge_log: list[str] = []
        chunks = split_text_for_tts(doc, max_chars=135, merge_log=merge_log)
        self.assertEqual(len(chunks), 1)
        self.assertIn("một.", chunks[0].text)
        self.assertIn("đọc đoạn văn", chunks[0].text)
        self.assertGreaterEqual(len(merge_log), 1)

    def test_tiny_orphan_merged_forward(self):
        merge_log: list[str] = []
        chunks = split_text_for_tts("một.\nđọc đoạn văn", max_chars=135, merge_log=merge_log)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text.strip(), "một. đọc đoạn văn")
        self.assertGreaterEqual(len(merge_log), 1)

    def test_paragraph_orphan_merged_forward(self):
        merge_log: list[str] = []
        raw = "word word word word word word word word word word.\nmột.\nmore text here please"
        chunks = split_text_for_tts(raw, max_chars=135, merge_log=merge_log)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[1].text.strip(), "một. more text here please")
        self.assertGreaterEqual(len(merge_log), 1)

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

    def test_read_text_file_normalizes_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_bytes("dòng một\r\ndòng hai\r\ndòng ba".encode("utf-8"))
            text = read_text_file(str(path))
        self.assertEqual(text, "dòng một\ndòng hai\ndòng ba")
        self.assertNotIn("\r", text)
        self.assertEqual(text.count("\n"), 2)


if __name__ == "__main__":
    unittest.main()
