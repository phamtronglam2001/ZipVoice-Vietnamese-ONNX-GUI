# Third-party model licenses

This folder contains neural network weights used for offline inference. **All weights in this repository are bundled for non-commercial use only.** See the project root [`LICENSE`](../LICENSE) and the matrix below.

> **Not legal advice.** If you use these weights outside research, education, or non-profit contexts, consult a lawyer and the original licensors.

---

## License matrix

| Artifact | Path | Source | License | Redistribute in this repo? | Required attribution |
|----------|------|--------|---------|------------------------------|----------------------|
| ZipVoice text encoder (INT4) | `onnx/text_encoder_int4.onnx` | Quantized export of hynt checkpoint | CC-BY-NC-SA-4.0 | **Yes** (same conditions) | Credit **Nguyen Thien Hy (`hynt`)**, link to HF model, note CC-BY-NC-SA-4.0, disclose AI-generated audio |
| ZipVoice flow-matching decoder (INT4) | `onnx/fm_decoder_int4.onnx` | Quantized export of hynt checkpoint | CC-BY-NC-SA-4.0 | **Yes** (same conditions) | Same as above |
| ZipVoice text encoder (INT8) | `onnx/text_encoder_int8.onnx` | Quantized export of hynt checkpoint | CC-BY-NC-SA-4.0 | **Yes** (same conditions) | Same as above |
| ZipVoice flow-matching decoder (INT8) | `onnx/fm_decoder_int8.onnx` | Quantized export of hynt checkpoint | CC-BY-NC-SA-4.0 | **Yes** (same conditions) | Same as above |
| Model config | `onnx/model.json` | From hynt checkpoint export | CC-BY-NC-SA-4.0 | **Yes** | Same as above |
| Token table | `onnx/tokens.txt` | From hynt checkpoint / k2-fsa ZipVoice | CC-BY-NC-SA-4.0 | **Yes** | Same as above |
| Vocos mel decoder (ONNX) | `vocoder/mel_spec_24khz.onnx` | User-bundled export from ZipVoice-Vietnamese-GUI (100 mel) · base weights [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) · architecture [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) | **MIT** (HF model cards & vocos code) | **Yes** | Include MIT copyright notice; cite Siuzdak et al. ([arXiv:2306.00814](https://arxiv.org/abs/2306.00814)) |
| Reference audio (30 voices) | `assets/sample_audio/*` | [contextboxai/ViZipvoice](https://huggingface.co/contextboxai/ViZipvoice) `audio/` (mp3 + sidecar `.txt`) | **Apache-2.0** | **Yes** | Credit ViZipVoice / [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) and original ZipVoice |

---

## CC-BY-NC-SA-4.0 (hynt ZipVoice weights)

The Vietnamese checkpoint and ONNX files derived from it are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**What this means for GitHub bundling:**

- **Redistribution is allowed** for **non-commercial** purposes (research, education, personal/non-profit sharing).
- You **must** give **attribution** to the original author (Nguyen Thien Hy / `hynt`).
- **ShareAlike:** adapted material (including ONNX exports) must be shared under **the same license** (CC-BY-NC-SA-4.0), not a more permissive one.
- **NonCommercial:** recipients must **not** use the weights or derivatives for commercial / monetized purposes.

This repository’s [`LICENSE`](../LICENSE) is aligned with NC use. Bundling hynt-derived ONNX weights here is **not blocked** by the license, provided attribution and SA terms are met.

**Training data** (PhoAudioBook, ViVoice, UEH) is referenced on the hynt model card; dataset terms are separate from these weight files.

---

## MIT (Vocos vocoder ONNX)

Bundled `mel_spec_24khz.onnx` is a **100-mel** export from ZipVoice-Vietnamese-GUI (aligned with ZipVoice `feat_dim`). Base PyTorch weights [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) and [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) source code are MIT.

Redistribution is permitted with copyright and permission notice preserved. Commercial use of the vocoder alone may be allowed under MIT, but **combined use with hynt ZipVoice ONNX remains non-commercial** because of the NC TTS weights.

---

## Apache-2.0 (k2-fsa ZipVoice code)

Inference scripts and architecture come from [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) (Apache-2.0). That license covers **code**, not the hynt fine-tuned **weights**, which remain CC-BY-NC-SA-4.0.

---

## Apache-2.0 (ViZipVoice reference audio)

The 30 bundled reference prompts under `assets/sample_audio/` are copied from the `audio/` folder of [contextboxai/ViZipvoice](https://huggingface.co/contextboxai/ViZipvoice) (each `.mp3` with a same-stem `.txt` transcript). Licensed under **Apache-2.0** per the ViZipVoice model card; retain attribution to ViZipVoice and the original ZipVoice project when redistributing.

---

## Suggested citation (research)

```bibtex
@misc{hynt2025zipvoicevi,
  author = {Nguyen Thien Hy},
  title = {ZipVoice-Vietnamese-2500h},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h}}
}
```

```bibtex
@article{siuzdak2023vocos,
  title={Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
  author={Siuzdak, Hubert},
  journal={arXiv preprint arXiv:2306.00814},
  year={2023}
}
```
