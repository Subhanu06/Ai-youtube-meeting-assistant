# 🎬 AI Video Assistant

**Transcribe. Summarise. Chat with your meetings — powered by AI.**

Turn any YouTube video or local audio/video file into a searchable, summarised, chat-ready knowledge base in minutes.

---

## ✨ Overview

AI Video Assistant is an end-to-end meeting/video intelligence pipeline built with **Streamlit**. Feed it a YouTube URL or a local file, and it will:

1. Download and chunk the audio
2. Transcribe speech to text using **Whisper**
3. Generate a title and summary
4. Extract action items, key decisions, and open questions
5. Build a **RAG (Retrieval-Augmented Generation)** index so you can chat directly with the transcript

All wrapped in a clean, dark-themed, real-time pipeline UI so you can watch each stage complete live.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔊 **Audio Ingestion** | Accepts YouTube links (via `yt-dlp`) or local audio/video files |
| ✂️ **Smart Chunking** | Splits long audio into manageable chunks for reliable transcription |
| 📝 **Speech-to-Text** | Fast, accurate transcription using `faster-whisper` |
| 🌐 **Multi-language Support** | English & Hinglish transcription modes |
| 🏷️ **Auto Title Generation** | Automatically names each session based on content |
| 📋 **AI Summarisation** | Condenses long transcripts into concise summaries |
| ✅ **Action Item Extraction** | Pulls out tasks, decisions, and open questions automatically |
| 🧠 **RAG-powered Chat** | Ask natural-language questions about your meeting and get grounded answers |
| 📊 **Live Pipeline Tracker** | Real-time sidebar + top-strip progress across all 6 processing stages |
| ⬇️ **Export** | Download the transcript or a full Markdown report |

---

## 🖥️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) with custom CSS (Syne + JetBrains Mono)
- **Audio Processing:** `yt-dlp`, `pydub`, `ffmpeg`
- **Transcription:** `faster-whisper` (English) + `Sarvam AI` STT (Hinglish/Indic languages)
- **LLM Orchestration:** `LangChain` + `Mistral AI`
- **Vector Store / RAG:** `ChromaDB` + `sentence-transformers`
- **Translation:** `deep-translator`
- **Export:** `reportlab`, `fpdf2`

---

## 📂 Project Structure

```
video-agent-ai/
├── app.py                     # Streamlit UI & pipeline orchestration
├── main.py                    # Entry point / CLI logic
├── core/
│   ├── extractor.py           # Action items, decisions, questions extraction
│   ├── rag_engine.py          # RAG chain build & Q&A
│   ├── summarize.py           # Summarisation & title generation
│   ├── transcriber.py         # Whisper-based transcription
│   └── vector_store.py        # Vector DB management
├── utils/
│   └── audio_processor.py     # Download, convert & chunk audio
├── vector_db/                 # Persisted vector store
├── downloads/                 # Downloaded/processed audio files
├── packages.txt               # System-level dependencies (ffmpeg)
├── requirements.txt           # Python dependencies
└── .env                       # API keys & environment config
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11
- [ffmpeg](https://ffmpeg.org/download.html) installed and available on your system `PATH`

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/video-agent-ai.git
cd video-agent-ai
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY="your_mistral_api_key"
WHISPER_MODEL="small"
SARVAM_API_KEY="your_sarvam_api_key"
SARVAM_STT_MODEL="saaras:v2.5"
```

| Variable | Description |
|---|---|
| `MISTRAL_API_KEY` | API key for Mistral AI (used for summarisation, extraction, and RAG chat) |
| `WHISPER_MODEL` | Whisper model size for transcription (`tiny`, `base`, `small`, `medium`, `large`) — `small` is a good balance of speed/accuracy |
| `SARVAM_API_KEY` | API key for [Sarvam AI](https://www.sarvam.ai/) speech-to-text (used for Hinglish/Indic language support) |
| `SARVAM_STT_MODEL` | Sarvam STT model version, e.g. `saaras:v2.5` |

> ⚠️ **Never commit your `.env` file.** Make sure it's listed in `.gitignore` before pushing to GitHub — it contains live API keys.

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🧭 Usage

1. Open the app in your browser (Streamlit will provide a local URL).
2. In the sidebar, paste a **YouTube URL** or a **local file path**.
3. Choose your transcription language (`english` / `hinglish`).
4. Click **⚡ Analyse** and watch the live pipeline process your video.
5. Once complete, explore:
   - 📋 Summary
   - 📝 Full transcript (downloadable)
   - ✅ Action items, 🔑 key decisions, ❓ open questions
   - 💬 Chat directly with your meeting via the RAG-powered chat box
6. Export a full Markdown report with one click.

---

## 🗺️ Pipeline Stages

```
🔊 Audio Processing  →  📝 Transcription  →  🏷️ Title Generation
        →  📋 Summarisation  →  🔍 Extraction  →  🧠 RAG Indexing
```

Each stage is tracked live in the sidebar with status, duration, and progress percentage.

---

## 🛠️ Roadmap

- [ ] Support for additional languages
- [ ] Speaker diarization
- [ ] Multi-file / batch processing
- [ ] Cloud deployment with proxy support for YouTube downloads
- [ ] Export to PDF/DOCX reports

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues) or open a pull request.

---

<p align="center">Built with ❤️ using Streamlit, LangChain & Whisper</p>
