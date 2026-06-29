import whisper
import nltk
import os
import shutil

nltk.download('punkt', quiet=True)

def get_ffmpeg_path():
    """Find and return ffmpeg executable path"""
    # Check if ffmpeg is in PATH
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print("✅ ffmpeg found in system PATH")
        return ffmpeg
    
    # Your specific WinGet path
    specific_path = r"C:\Users\sujth\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
    if os.path.exists(specific_path):
        print("✅ Using ffmpeg from WinGet location")
        return specific_path
    
    # Common fallback paths
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Found ffmpeg at: {path}")
            return path
    
    print("⚠️ ffmpeg not found! Please install ffmpeg.")
    return None


# Load Whisper model once
model = whisper.load_model("base")

def speech_to_text(audio_path: str, language: str = "en") -> dict:
    try:
        ffmpeg_path = get_ffmpeg_path()
        if ffmpeg_path:
            # Add to PATH
            os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
            os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
        
        if not os.path.exists(audio_path):
            return {"success": False, "error": f"Audio file not found: {audio_path}"}
        
        # Transcribe
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
        return {"success": False, "error": str(e)}


def get_word_count(text: str) -> int:
    return len(text.split())


def get_speaking_rate(text: str, duration_seconds: float) -> float:
    words = get_word_count(text)
    minutes = duration_seconds / 60
    return round(words / minutes, 2) if minutes > 0 else 0


def filler_word_ratio(text: str) -> float:
    fillers = ["um", "uh", "like", "you know", "basically", "actually", "literally", "so", "right", "okay"]
    words = text.lower().split()
    total = len(words)
    filler_count = sum(1 for word in words if word in fillers)
    return round(filler_count / total, 4) if total > 0 else 0
