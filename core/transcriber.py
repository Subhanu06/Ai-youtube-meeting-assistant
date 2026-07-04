from faster_whisper import WhisperModel
import os
import requests
from pydub import AudioSegment

# ---------------- CONFIG ---------------- #

SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None

# ---------------- WHISPER LOADER ---------------- #

def load_model():
    global _model

    if _model is None:
        print(f"Loading faster-whisper model: {WHISPER_MODEL}")

        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",          # change to "cuda" if GPU available
            compute_type="int8"    # fastest CPU mode
        )

        print("Whisper loaded.")

    return _model


# ---------------- WHISPER TRANSCRIBE ---------------- #

def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    segments, info = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    text = " ".join(segment.text for segment in segments)
    return text.strip()


# ---------------- SARVAM API CALL ---------------- #

def _send_to_sarvam(piece_path: str) -> str:
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {
            "file": (os.path.basename(piece_path), f, "audio/wav")
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"❌ Sarvam error {response.status_code}")
        print(response.text)
        response.raise_for_status()

    return response.json().get("transcript", "")


# ---------------- SARVAM CHUNKING ---------------- #

def transcribe_chunk_sarvam(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY missing")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start:start + piece_ms]

        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"→ Sarvam chunk {i+1}")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


# ---------------- ROUTER ---------------- #

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


# ---------------- MAIN PIPELINE ---------------- #

def transcribe_all(chunks: list, language: str = "english") -> str:
    full_text = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Faster Whisper"
    print(f"Using {engine}")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing {i+1}/{len(chunks)}")

        text = transcribe_chunk(chunk, language)
        full_text += text + " "

    return full_text.strip()

print("WHISPER_MODEL:", WHISPER_MODEL)
print("SARVAM_MODEL:", SARVAM_MODEL)
print("API KEY loaded:", bool(SARVAM_API_KEY))