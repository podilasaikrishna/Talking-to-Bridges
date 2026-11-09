import os
import threading

# Fix for OpenMP duplicate library error on Windows/Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel

_model = None
_device = None
_lock = threading.Lock()


def _load(device: str):
    """Load the Whisper model on the given device ("cuda" or "cpu")."""
    global _model, _device
    if device == "cuda":
        _model = WhisperModel("base", device="cuda", compute_type="float16")
    else:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    _device = device
    return _model


def get_model():
    """Lazy-load the model on first use (GPU if possible, otherwise CPU)."""
    if _model is None:
        print("Loading Speech-to-Text Model (faster-whisper, base)...")
        try:
            _load("cuda")
        except Exception as e:
            print(f"GPU load failed ({e}). Falling back to CPU...")
            _load("cpu")
    return _model


def _run(model, audio_file_path: str) -> str:
    # NOTE: `segments` is a lazy generator - the real work (and any CUDA/cuBLAS/cuDNN
    # error on Windows) happens while it is consumed, so it must be consumed in here.
    segments, _info = model.transcribe(audio_file_path, beam_size=5, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe(audio_file_path):
    """What goes in: audio file path. What comes out: the spoken text ("" if silence)."""
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    with _lock:
        model = get_model()
        try:
            return _run(model, audio_file_path)
        except Exception as e:
            if _device == "cuda":
                print(f"GPU transcription failed ({e}). Retrying on CPU...")
                return _run(_load("cpu"), audio_file_path)
            raise


# Quick test if you run this file directly
if __name__ == "__main__":
    test_audio = "speech/sample_test_audio.wav"
    if os.path.exists(test_audio):
        print("Result:", transcribe(test_audio))
    else:
        print(f"Could not find {test_audio}")
