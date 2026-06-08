"""Minimal tests for normalize pipeline chaining."""

from __future__ import annotations



import tempfile

import unittest

from pathlib import Path



import numpy as np



from audio.post_process import (
    apply_chunk_wave_pauses,
    join_tts_audio_chunks,
    prepend_leading_pause,
)
from text.chunking import TtsChunk, format_chunks_preview, split_text_for_tts
from text.io import read_text_file
from text.normalizers import (
    build_normalize_pipeline,
    normalize_text_pipeline,
)
from text.normalizers.dot_newline import dot_space_to_newline
from text.normalizers.period_linebreak import (
    brackets_to_newlines,
    join_soft_breaks,
    newline_sentence_boundary,
    prepare_tts_structure,
)
from text.pipeline import normalize_full_document, post_process_text





class NormalizePipelineTests(unittest.TestCase):

    def test_dot_space_empty_pipeline_no_conversion(self):
        raw = "một. đọc"
        doc = normalize_full_document(raw, [])
        self.assertNotIn("\n", doc)
        self.assertEqual(doc, "một. đọc")

    def test_dot_space_pipeline_step_converts(self):
        raw = "một. đọc"
        pipe = build_normalize_pipeline(["dot_newline"])
        out = normalize_text_pipeline(raw, pipe)
        self.assertEqual(out, "một.\nđọc")

    def test_dot_space_direct(self):
        self.assertEqual(dot_space_to_newline("một. đọc"), "một.\nđọc")

    def test_dot_space_preserves_decimal(self):
        raw = "3.14 số"
        self.assertEqual(dot_space_to_newline(raw), raw)
        doc = normalize_full_document(raw, [])
        self.assertNotIn("\n", doc)
        self.assertIn("3.14", doc)

    def test_dot_space_skips_digit_before_dot(self):
        raw = "3. 14 số"
        self.assertEqual(dot_space_to_newline(raw), raw)

    def test_vieneu_period_break_preserves_dot_newlines(self):
        raw = "một. đọc đoạn"
        pipe = build_normalize_pipeline(["dot_newline", "vieneu", "period_break"])
        out = normalize_text_pipeline(raw, pipe)
        self.assertIn("\n", out)
        self.assertIn("một.", out)
        self.assertIn("đọc", out)

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

        self.assertIn("\n", chunks[0].text)

        self.assertIsNone(chunks[0].merged_boundary_pause)

        self.assertEqual(chunks[0].merged_prefix_len, 0)

        self.assertEqual(chunks[0].leading_pause, 0.0)

        self.assertEqual(chunks[0].pause_after, 0.0)

        self.assertTrue(any("Gộp" in m for m in merge_log))



    def test_micro_orphan_merged_forward_with_newline(self):

        """Sub-min_chars line merged into next chunk via \\n (one generate call)."""

        merge_log: list[str] = []

        chunks = split_text_for_tts("một.\nđọc đoạn văn", max_chars=135, merge_log=merge_log)

        self.assertEqual(len(chunks), 1)

        self.assertEqual(chunks[0].text.strip(), "một.\nđọc đoạn văn")

        self.assertIsNone(chunks[0].merged_boundary_pause)

        self.assertEqual(chunks[0].merged_prefix_len, 0)

        self.assertEqual(chunks[0].leading_pause, 0.0)

        self.assertEqual(chunks[0].pause_after, 0.0)

        self.assertTrue(any("Gộp" in m for m in merge_log))



    def test_micro_orphan_merged_into_next_block(self):

        merge_log: list[str] = []

        raw = "word word word word word word word word word word.\nmột.\nmore text here please"

        chunks = split_text_for_tts(raw, max_chars=135, min_chars=12, merge_log=merge_log)

        self.assertEqual(len(chunks), 2)

        self.assertIn("một.", chunks[1].text)

        self.assertIn("more text here please", chunks[1].text)

        self.assertIn("\n", chunks[1].text)

        self.assertIsNone(chunks[1].merged_boundary_pause)

        self.assertEqual(chunks[1].merged_prefix_len, 0)

        self.assertEqual(chunks[1].leading_pause, 0.0)

        self.assertTrue(any("Gộp" in m for m in merge_log))



    def test_micro_orphan_line_merges_with_newline(self):

        """Short lines below min_chars collapse to one synthesis unit with \\n."""

        merge_log: list[str] = []

        for raw in ("một.\nví dụ", "một.\n\nví dụ"):

            with self.subTest(raw=raw):

                chunks = split_text_for_tts(raw, max_chars=135, merge_log=merge_log)

                self.assertEqual(len(chunks), 1, raw)

                self.assertEqual(chunks[0].text.strip(), "một.\nví dụ")

                self.assertIsNone(chunks[0].merged_boundary_pause)

                self.assertEqual(chunks[0].merged_prefix_len, 0)

                self.assertEqual(chunks[0].leading_pause, 0.0)

                self.assertEqual(chunks[-1].pause_after, 0.0)



    def test_join_prepends_leading_pause_before_chunk(self):

        sr = 24000

        chunks = [

            TtsChunk(text="a", pause_after=0.0, leading_pause=0.5),

        ]

        speech = np.ones(sr, dtype=np.float32)

        out = join_tts_audio_chunks([speech], chunks, sr)

        self.assertEqual(len(out), sr + int(0.5 * sr))



    def test_apply_chunk_wave_pauses_leading_only(self):

        sr = 24000

        speech = np.ones(sr, dtype=np.float32)

        chunk = TtsChunk(

            text="một. ví dụ",

            leading_pause=0.2,

            merged_boundary_pause=0.5,

            merged_prefix_len=4,

        )

        out = apply_chunk_wave_pauses(speech, chunk, sr)

        leading_gap = int(0.2 * sr)

        self.assertEqual(len(out), sr + leading_gap)

        self.assertTrue(np.all(out[:leading_gap] == 0.0))



    def test_prepend_leading_pause_helper(self):

        sr = 24000

        speech = np.ones(100, dtype=np.float32)

        out = prepend_leading_pause(speech, 0.1, sr)

        self.assertEqual(len(out), 100 + int(0.1 * sr))



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



    def test_brackets_to_newlines(self):

        self.assertEqual(brackets_to_newlines("mẫu (mẹ)"), "mẫu\nmẹ")

        self.assertEqual(brackets_to_newlines("a [b] c"), "a\nb\nc")

        self.assertEqual(brackets_to_newlines("x {y} z"), "x\ny\nz")

        self.assertEqual(brackets_to_newlines("mẫu (mẹ)"), "mẫu\nmẹ")



    def test_period_break_brackets_in_pipeline(self):

        raw = "mẫu (mẹ) đọc"

        pipe = build_normalize_pipeline(["period_break"])

        out = normalize_text_pipeline(raw, pipe)

        self.assertIn("mẫu", out)

        self.assertIn("mẹ", out)

        self.assertIn("\n", out)

        self.assertNotIn("(", out)

        self.assertNotIn(")", out)



    def test_min_chunk_chars_merges_medium_chunks(self):

        """Chunks below min_chars (default 70) merge even when above noise floor."""

        s1 = "a" * 45 + "."

        s2 = "b" * 45 + "."

        raw = f"{s1}\n\n{s2}"

        merge_log: list[str] = []

        chunks = split_text_for_tts(

            raw,

            max_chars=135,

            min_chars=70,

            merge_log=merge_log,

        )

        self.assertEqual(len(chunks), 1)

        self.assertGreaterEqual(len(chunks[0].text), 70)

        self.assertIn("\n", chunks[0].text)

        self.assertTrue(any("Gộp" in m for m in merge_log))



    def test_micro_chunk_merge_uses_newline_join(self):

        """Sub-min_chars units merge with \\n (not space) before one generate() call."""

        merge_log: list[str] = []

        raw = "một.\nví dụ ngắn"

        chunks = split_text_for_tts(raw, max_chars=135, min_chars=12, merge_log=merge_log)

        self.assertEqual(len(chunks), 1)

        self.assertEqual(chunks[0].text.strip(), "một.\nví dụ ngắn")

        self.assertNotIn("một. ví", chunks[0].text)

        self.assertTrue(any("Gộp" in m for m in merge_log))



    def test_regular_chunks_stay_separate_use_pause_not_newline(self):

        """Chunks >= min_chars stay separate synthesis units; gaps via pause_after."""

        p1 = "word " * 15 + "first."

        p2 = "word " * 15 + "second."

        chunks = split_text_for_tts(f"{p1}\n\n{p2}", max_chars=135, min_chars=70)

        self.assertEqual(len(chunks), 2)

        self.assertNotIn("\n", chunks[0].text)

        self.assertNotIn("\n", chunks[1].text)

        self.assertGreater(chunks[0].pause_after, 0.0)

        self.assertEqual(chunks[1].pause_after, 0.0)

        sr = 24000

        speech = np.ones(sr, dtype=np.float32)

        out = join_tts_audio_chunks([speech, speech], chunks, sr)

        gap = int(chunks[0].pause_after * sr)

        self.assertEqual(len(out), 2 * sr + gap)



    def test_merge_avoids_double_newline(self):

        """Re-merging parts that already contain newlines must not stack \\n\\n."""

        merge_log: list[str] = []

        raw = "một.\n\nví dụ"

        chunks = split_text_for_tts(raw, max_chars=135, merge_log=merge_log)

        self.assertEqual(len(chunks), 1)

        self.assertNotIn("\n\n", chunks[0].text)

        self.assertEqual(chunks[0].text.strip(), "một.\nví dụ")



    def test_min_chunk_chars_zero_disables_merge_floor(self):

        s1 = "word " * 8

        s2 = "more " * 8

        raw = f"{s1.strip()}.\n\n{s2.strip()}."

        chunks = split_text_for_tts(raw, max_chars=135, min_chars=12)

        self.assertGreaterEqual(len(chunks), 1)



    def test_format_chunks_preview_newline_tag_and_pause_metadata(self):
        merge_log: list[str] = []
        chunks = split_text_for_tts(
            "một.\nví dụ đoạn văn",
            max_chars=135,
            min_chars=12,
            pause_paragraph=0.65,
            merge_log=merge_log,
        )
        preview = format_chunks_preview(chunks, show_micro_merge=True)
        self.assertIn("[NL]", preview)
        self.assertIn("pause_after=", preview)
        self.assertIn("Chunk 1/", preview)
        self.assertIn("một.", preview)
        self.assertIn("ví dụ đoạn văn", preview)
        if "\n" in chunks[0].text:
            self.assertIn("[micro-merged]", preview)

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

