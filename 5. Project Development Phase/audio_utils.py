import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io

def extract_audio_features(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms_energy = float(np.mean(librosa.feature.rms(y=y)))
        zero_crossing = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        
        intervals = librosa.effects.split(y, top_db=30)
        speech_duration = sum([(e-s)/sr for s,e in intervals])
        pause_duration = duration - speech_duration
        pause_ratio = round(pause_duration / duration, 4) if duration > 0 else 0

        return {
            "success": True,
            "duration": round(duration, 2),
            "rms_energy": round(rms_energy, 4),
            "zero_crossing_rate": round(zero_crossing, 4),
            "pause_ratio": pause_ratio,
            "speech_duration": round(speech_duration, 2),
            "pause_duration": round(pause_duration, 2),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def save_waveform(audio_path: str) -> bytes:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(figsize=(10, 3))
        times = np.linspace(0, len(y)/sr, len(y))
        ax.plot(times, y, color='#00b4d8', linewidth=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title("Audio Waveform")
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                   facecolor=fig.get_facecolor())
        plt.close()
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        return None
