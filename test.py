from core.transcriber import transcribe_all
from utils.audio_processor import process_input

chunks = process_input("https://youtu.be/5DfVeofl37o?si=RDf4xAQSFo2nwB3s")
print(transcribe_all(chunks))