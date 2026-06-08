# Demo assets for README

Files here are linked from the root README (screenshot + HTML table with **play** links).

Pattern follows [gwen-tts](https://github.com/ggroup-ai-lab/gwen-tts): relative path + `<a href="...">play</a>` — GitHub opens the file with its built-in player.

| File | Purpose |
|------|---------|
| `../screenshot.png` | Gradio GUI screenshot |
| `ref_voice.mp3` | Reference voice (`play` link in README) |
| `output.wav` | Synthesized output (`play` link in README) |
| `sample_text.txt` | Source text — keep in sync with README italic block |

## Regenerate

1. Screenshot → `docs/screenshot.png`
2. Reference → copy e.g. `assets/sample_audio/Bá-Vinh.mp3` → `ref_voice.mp3`
3. Run TTS with `sample_text.txt` → `output.wav`
4. Update README italic text if `sample_text.txt` changes
5. Commit and push (`docs/screenshot.png`, `docs/demo/*.mp3` are whitelisted in `.gitignore`)
