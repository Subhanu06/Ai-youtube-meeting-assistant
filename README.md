# 🎬 AI Video Assistant

**Transcribe. Summarise. Chat with your meetings — powered by AI.**

Turn any YouTube video or local audio/video file into a searchable, summarised, chat-ready knowledge base in minutes.

---

## ⚠️ Running on Free Tier

> This project currently runs on **free-tier APIs and hosting** (Whisper API, OpenAI API, and free Streamlit hosting). Because of rate limits and cold-start delays on free tiers, **processing can take longer than expected** — especially for longer videos or during peak usage. Please be patient while a job is running; it hasn't hung, it's just working within free-tier limits. Upgrading to paid tiers will significantly speed up transcription, embedding, and chat response times.

---

## ⬆️ Want Better Performance? Upgrade the Stack

This project ships configured to run **free and local-first** (local `faster-whisper` for transcription, `sentence-transformers` for embeddings, and Mistral AI for chat). That keeps it free to run, but it also means it can be slow — especially on longer videos or busy free-tier quotas.

If you want faster, higher-quality results, you can swap in paid APIs:

- **Whisper API (OpenAI)** — Replace local `faster-whisper` transcription with OpenAI's hosted Whisper API for faster, more accurate transcription without running a model on your own machine.
- **OpenAI Embeddings** — Swap `sentence-transformers` for OpenAI's embedding API in the vector store for higher-quality retrieval in the RAG chat.
- **OpenAI Chat** — Use OpenAI's chat completion API instead of (or alongside) Mistral AI for the RAG-powered Q&A, which can give faster and more consistent answers.

These are drop-in upgrades — see the `.env` variables below to enable them. You'll need your own OpenAI API key and should expect standard OpenAI usage costs once enabled.

---

## ✨ Overview

AI Video Assistant is an end-to-end meeting/video intelligence pipeline built with **Streamlit**. Feed it a YouTube URL or a local file, and it will:

1. Download and chunk the audio
2. Transcribe speech to text using **Whisper** (local `faster-whisper` by default, or the **OpenAI Whisper API** if you upgrade)
3. Generate a title and summary
4. Extract action items, key decisions, and open questions
5. Build a **RAG (Retrieval-Augmented Generation)** index — using `sentence-transformers` by default, or **OpenAI embeddings** if you upgrade — so you can chat directly with the transcript

All wrapped in a clean, dark-themed, real-time pipeline UI so you can watch each stage complete live.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔊 **Audio Ingestion** | Accepts YouTube links (via `yt-dlp`) or local audio/video files |
| ✂️ **Smart Chunking** | Splits long audio into manageable chunks for reliable transcription |
| 📝 **Speech-to-Text** | Fast, accurate transcription using `faster-whisper` locally — **upgradeable** to the hosted OpenAI Whisper API |
| 🌐 **Multi-language Support** | English & Hinglish transcription modes |
| 🏷️ **Auto Title Generation** | Automatically names each session based on content |
| 📋 **AI Summarisation** | Condenses long transcripts into concise summaries |
| ✅ **Action Item Extraction** | Pulls out tasks, decisions, and open questions automatically |
| 🧠 **RAG-powered Chat** | Ask natural-language questions about your meeting and get grounded answers via Mistral AI — **upgradeable** to OpenAI embeddings/chat |
| 📊 **Live Pipeline Tracker** | Real-time sidebar + top-strip progress across all 6 processing stages |
| ⬇️ **Export** | Download the transcript or a full Markdown report |

---

## 🖥️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) with custom CSS (Syne + JetBrains Mono)
- **Audio Processing:** `yt-dlp`, `pydub`, `ffmpeg`
- **Transcription:** `faster-whisper` (local, English, default) + `Sarvam AI` STT (Hinglish/Indic languages) — *upgradeable to OpenAI Whisper API*
- **LLM Orchestration:** `LangChain` + `Mistral AI` (default) — *upgradeable to OpenAI API*
- **Vector Store / RAG:** `ChromaDB` + `sentence-transformers` (default) — *upgradeable to OpenAI Embeddings API*
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
│   ├── rag_engine.py          # RAG chain build & Q&A (Mistral / OpenAI)
│   ├── summarize.py           # Summarisation & title generation
│   ├── transcriber.py         # Whisper-based transcription (local + API)
│   └── vector_store.py        # Vector DB management (sentence-transformers / OpenAI embeddings)
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

# --- Optional: upgrade to paid OpenAI APIs for faster/higher-quality results ---
OPENAI_API_KEY="your_openai_api_key"
USE_WHISPER_API="false"          # set to "true" to use OpenAI's hosted Whisper API instead of local faster-whisper
USE_OPENAI_EMBEDDINGS="false"    # set to "true" to use OpenAI embeddings instead of sentence-transformers
USE_OPENAI_CHAT="false"          # set to "true" to use OpenAI chat completions instead of Mistral AI
```

| Variable | Description |
|---|---|
| `MISTRAL_API_KEY` | API key for Mistral AI (used for summarisation, extraction, and RAG chat by default) |
| `WHISPER_MODEL` | Whisper model size for local transcription (`tiny`, `base`, `small`, `medium`, `large`) — `small` is a good balance of speed/accuracy |
| `SARVAM_API_KEY` | API key for [Sarvam AI](https://www.sarvam.ai/) speech-to-text (used for Hinglish/Indic language support) |
| `SARVAM_STT_MODEL` | Sarvam STT model version, e.g. `saaras:v2.5` |
| `OPENAI_API_KEY` | *(Optional upgrade)* API key for OpenAI, needed if you enable the Whisper API, OpenAI embeddings, or OpenAI chat below |
| `USE_WHISPER_API` | *(Optional upgrade)* Set `true` to use OpenAI's hosted Whisper API instead of local `faster-whisper` |
| `USE_OPENAI_EMBEDDINGS` | *(Optional upgrade)* Set `true` to use OpenAI embeddings instead of `sentence-transformers` for the vector store |
| `USE_OPENAI_CHAT` | *(Optional upgrade)* Set `true` to use OpenAI chat completions instead of Mistral AI for RAG-powered chat |

> 💡 These OpenAI-based upgrades are **optional**. If you don't set them, the app runs fully on the free/local stack (Mistral AI, local Whisper, sentence-transformers) — just slower than the paid alternatives.

> ⚠️ **Never commit your `.env` file.** Make sure it's listed in `.gitignore` before pushing to GitHub — it contains live API keys.

> 🐢 **Note:** All API integrations (Whisper API, OpenAI embeddings/chat, Sarvam AI) are currently configured on **free-tier plans**. Expect slower processing times, occasional rate-limit waits, or queuing during heavy use — this is expected behavior on free tiers, not a bug.

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
   - ⏳ By default the app runs on free/local models, so larger files or busy periods may take a few extra minutes — the pipeline tracker will keep you updated on progress. See [Want Better Performance?](#-want-better-performance-upgrade-the-stack) if you'd like to speed this up with paid APIs.
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
- [ ] Move off free-tier APIs / add paid-tier fallback for faster processing

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues) or open a pull request.

---

<p align="center">Built with ❤️ using Streamlit, LangChain & Whisper</p>
