# Demo assets for README

Files here are embedded directly in the root README (screenshot, inline audio players, sample text).

| File | Purpose |
|------|---------|
| `../screenshot.png` | Gradio GUI screenshot |
| `ref_voice.mp3` | Reference voice (HTML `<audio>` player in README) |
| `output.wav` | Synthesized output (HTML `<audio>` player in README) |
| `sample_text.txt` | Source text for README blockquote (keep in sync manually) |

## Regenerate

1. Screenshot: capture Gradio → save as `docs/screenshot.png`
2. Reference: copy a bundled voice, e.g. `assets/sample_audio/Bá-Vinh.mp3` → `ref_voice.mp3`
3. Run TTS with `sample_text.txt` → save as `output.wav`
4. Commit and push — `.gitignore` whitelists `docs/screenshot.png` and `docs/demo/*.mp3`
