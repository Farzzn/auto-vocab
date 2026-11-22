# Auto-Vocab (MPC-HC Vocabulary Assistant)

A lightweight Windows desktop companion for **MPC-HC 64-bit** that detects when you pause a video, reads the active subtitle (embedded text tracks, external subtitle files, or image-based/hardcoded subtitles via Windows OCR), filters out common stopwords and everyday words, and displays vocabulary definitions on the right side of the paused frame without stealing keyboard focus.

## Features

- **Zero-click pause overlay:** Automatically docks on the right edge of MPC-HC (above the subtitle region) when you pause with `Space`, and disappears when playback resumes. Uses `WS_EX_NOACTIVATE` so `Space` always resumes the video directly.
- **Embedded, external & OCR subtitle support:**
  - Automatically probes and extracts embedded text subtitle tracks (`.mkv`, `.mp4`, etc.) via `ffmpeg`/`ffprobe` and caches them locally.
  - Supports external `.srt`, `.ass`, `.ssa`, and `.vtt` subtitle files.
  - Falls back automatically to **Windows built-in OCR** (`Windows.Media.Ocr`) with a second-pass contrast boost for image-based subtitles (**PGS / VobSub**) and burned-in subtitles.
- **Smart filtering + live Rarity Slider:** Strips stopwords and contractions, lemmatizes words with POS hints, and filters by `wordfreq` Zipf score. Drag the **Word Rarity Filter** slider while paused to show rarer or more common words in real time.
- **Offline-first dictionary + on-demand Gemini AI:**
  - Instant `<5ms` offline definitions from NLTK WordNet (`0` API calls when pausing).
  - Words missing from WordNet show **`Offline miss`** with an on-demand **`✨ Ask Gemini`** button (and offline cards include **`✨ Context`**) powered by the `google-genai` Interactions API (`gemini-3.5-flash-lite` by default).
  - Sends the full subtitle sentence as context, infers likely **OCR/spelling typos** (e.g., `wpmer` → `women`), and returns a concise meaning, synonym, and natural usage example sentence (cached locally in `data/llm_cache.json`).
- **Save vocabulary to CSV:** Click **`+ Save`** on any word card to export the word, lemma, meaning, sentence context, video filename, and timestamp to `data/saved_vocab.csv`.
- **Start with Windows:** Toggle **Start with Windows** directly from the system tray menu (`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`).

## Requirements

- Windows 10 / 11 (64-bit)
- **MPC-HC 64-bit** (`MediaPlayerClassicW`)
- **Python 3.11+**
- **`ffmpeg` and `ffprobe`** available on `PATH` (for embedded subtitle extraction)
- *(Optional)* **Gemini API key** via `GEMINI_API_KEY` environment variable, `.env`, or the in-app **Settings...** dialog

## Setup (Fresh Clone)

1. **Create a virtual environment and install dependencies:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\pip.exe install -r requirements.txt
   .\.venv\Scripts\python.exe setup_nltk.py
   ```

2. **Enable MPC-HC Integration (one-time, close MPC-HC first):**
   ```powershell
   .\.venv\Scripts\python.exe setup_mpc_integration.py
   ```
   *(Or right-click the tray icon → **Enable MPC-HC Integration** while MPC-HC is closed.)*  
   This enables MPC-HC's built-in localhost status endpoint (`127.0.0.1:13579`, localhost-only) so you never need to run a separate web server.

   > **Note:** Make sure **Direct3D Fullscreen** is unchecked in MPC-HC (`Options → Playback → Output`) so overlays can appear over fullscreen playback.

3. **Run the Assistant:**
   - Double-click `run_silent.vbs` (runs silently in the Windows system tray with no console window)
   - Or run `run.bat` for console output.
   - Right-click the **"V"** tray icon → check **Start with Windows** to launch automatically on boot.

## Running Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
