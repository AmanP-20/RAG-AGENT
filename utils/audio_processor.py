import os
from typing import List
from pydub import AudioSegment
import yt_dlp

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Downloads YouTube audio and normalizes it to a 16kHz Mono WAV 
    file optimized directly for WhisperAI via yt_dlp/FFmpeg.
    """
    output_template = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # Forces FFmpeg to output 16000Hz, Mono channel directly
        "postprocessor_args": [
            "-ar", "16000",
            "-ac", "1"
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            base_path = os.path.splitext(ydl.prepare_filename(info))[0]
            return f"{base_path}.wav"
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        raise


def normalize_local_audio(input_path: str) -> str:
    """
    Converts a local audio/video file into a 16kHz Mono WAV file using pydub.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Local file not found: {input_path}")

    base_path, _ = os.path.splitext(input_path)
    output_path = f"{base_path}_normalized.wav"

    try:
        audio = AudioSegment.from_file(input_path)
        # Enforce Whisper-compatible specs: Single channel (Mono), 16000Hz
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error processing local file {input_path}: {e}")
        raise


def chunk_audio(wav_path: str, chunk_minute: int = 10) -> List[str]:
    """
    Splits a WAV audio file into smaller chunks of specified minute intervals.
    """
    try:
        audio = AudioSegment.from_wav(wav_path)
        chunk_ms = chunk_minute * 60 * 1000
        chunks = []
        
        base_path, ext = os.path.splitext(wav_path)
        
        for i, start in enumerate(range(0, len(audio), chunk_ms)):
            chunk = audio[start : start + chunk_ms]
            chunk_path = f"{base_path}_chunk_{i}{ext}"  # Cleans path to: file_chunk_0.wav
            
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
            
        return chunks
    except Exception as e:
        print(f"Error during audio chunking: {e}")
        raise


def process_input(source: str) -> List[str]:
    """
    Orchestrates the ingestion of either a YouTube URL or a local file,
    ensures it's formatted for Whisper, and chunks it.
    """
    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading and formatting audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting and formatting to WAV...")
        wav_path = normalize_local_audio(source)

    print(f"Splitting audio into chunks...")
    chunks = chunk_audio(wav_path)

    print(f"Audio processing complete. {len(chunks)} chunk(s) created.")
    return chunks


if __name__ == "__main__":
    sample_url = "https://youtu.be/XJtIutUX3DU?si=OIy9dJMTbAVk8txP"
    try:
        processed_chunks = process_input(sample_url)
        print("Generated Chunks:", processed_chunks)
    except Exception as main_error:
        print(f"Pipeline Execution Failed: {main_error}")