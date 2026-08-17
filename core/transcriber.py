import os
import whisper
import requests
from pydub import AudioSegment

# --- Configuration & Constants ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Sarvam Config
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

# Sarvam's sync API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

_model = None


# --- Model Loading ---
def load_whisper_model():
    """Lazily load the Whisper model into memory."""
    global _model
    if _model is None:
        print(f"Loading Whisper model '{WHISPER_MODEL}' into memory...")
        # If not on disk, it downloads once. Otherwise, it just loads from cache.
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper Model Loaded Successfully.")
    return _model


# --- Whisper Processing ---
def transcribe_chunk_whisper(chunk_path: str, task: str = "transcribe") -> str:
    """Process a chunk through the local Whisper model."""
    model = load_whisper_model()
    result = model.transcribe(chunk_path, task=task)
    return result["text"]


# --- Sarvam AI Processing ---
def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts <=30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment or .env file.")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            # Always clean up the temporary split file
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


# --- Main Routing Logic ---
def transcribe_chunk(chunk_path: str, language: str = "english", task: str = "transcribe") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  -> Whisper (local model)
    - hinglish/hindi -> Sarvam (translates to English while transcribing)
    """
    target_lang = language.lower()
    
    if target_lang in ["hinglish", "hindi"]:
        return transcribe_chunk_sarvam(chunk_path)
    
    # Fall back to local Whisper for English (or if you want to use Whisper's own translation task)
    return transcribe_chunk_whisper(chunk_path, task=task)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """
    Iterate through all audio chunks and return a single combined transcript.
    Utilizes list joining for faster processing.
    """
    transcripts = []
    engine = "Sarvam AI" if language.lower() in ["hinglish", "hindi"] else "Whisper"
    
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}: {os.path.basename(chunk)}")
        
        # Process the chunk through the router
        text = transcribe_chunk(chunk, language=language)
        transcripts.append(text.strip())

    print("Transcription complete.")
    
    # Joining a list is faster and cleaner than string concatenation (+=) in a loop
    return " ".join(transcripts)