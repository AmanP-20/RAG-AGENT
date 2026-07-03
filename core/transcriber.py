import os
import whisper

# Fixed typo to WHISPER_MODEL
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

def load_whisper_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model '{WHISPER_MODEL}' into memory...")
        # If not on disk, it downloads once. Otherwise, it just loads from cache.
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper Model Loaded Successfully.")
    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    model = load_whisper_model()
    task = "translate" if translate else "transcribe"
    result = model.transcribe(chunk_path, task=task)
    return result["text"]


def transcribe_all(chunks: list, translate: bool = False) -> str:
    transcripts = []
    for chunk in chunks:
        print(f"Transcribing: {os.path.basename(chunk)}")
        text = transcribe_chunk(chunk, translate=translate)
        transcripts.append(text.strip())

    # Joining a list is faster and cleaner than string concatenation (+=) in a loop
    return " ".join(transcripts)