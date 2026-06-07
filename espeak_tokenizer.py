"""
Minimal Espeak tokenizer for ZipVoice Vietnamese ONNX inference.
Vendored from k2-fsa/ZipVoice (Apache 2.0) — no jieba/lhotse/pypinyin imports.
"""
from __future__ import annotations

import logging
from functools import reduce
from typing import Dict, List, Optional

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nInstall: pip install piper_phonemize -f "
        "https://k2-fsa.github.io/icefall/piper_phonemize.html"
    ) from ex

logger = logging.getLogger("zipvoice_onnx_gui")


class EspeakTokenizer:
    def __init__(self, token_file: str, lang: str = "vi"):
        self.lang = lang
        self.token2id: Dict[str, int] = {}
        with open(token_file, encoding="utf-8") as f:
            for line in f:
                token, tid = line.rstrip().split("\t")
                self.token2id[token] = int(tid)
        self.pad_id = self.token2id["_"]
        self.vocab_size = len(self.token2id)

    def g2p(self, text: str) -> List[str]:
        try:
            tokens = phonemize_espeak(text, self.lang)
            return reduce(lambda x, y: x + y, tokens)
        except Exception as ex:
            logger.warning("Espeak tokenization failed (%s): %s", self.lang, ex)
            return []

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        return [self.g2p(t) for t in texts]

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        out: List[List[int]] = []
        for tokens in tokens_list:
            ids = [self.token2id[t] for t in tokens if t in self.token2id]
            out.append(ids)
        return out

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))
