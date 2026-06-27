import whisper
import nltk
nltk.download('punkt', quiet=True)

model = whisper.load_model("base")

def speech_to_text(audio_path: str, language: str = "en") -> dict:
    try:
        if language == "te":
            result = model.transcribe(audio_path, language="te", task="transcribe")
        else:
            result = model.transcribe(audio_path, language="en", task="transcribe")
        return {
            "success": True,
            "text": result["text"].strip(),
            "language": language,
            "segments": result.get("segments", [])
        }
    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}

def get_word_count(text: str) -> int:
    return len(text.split())

def get_speaking_rate(text: str, duration_seconds: float) -> float:
    words = get_word_count(text)
    minutes = duration_seconds / 60
    return round(words / minutes, 2) if minutes > 0 else 0

def filler_word_ratio(text: str) -> float:
    fillers = ["um","uh","like","you know","basically",
               "actually","literally","so","right","okay"]
    words = text.lower().split()
    total = len(words)
    filler_count = sum(words.count(f) for f in fillers)
    return round(filler_count / total, 4) if total > 0 else 0
