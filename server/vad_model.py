"""Chargement du vrai modèle Silero VAD (ONNX, CPU) — hors chemin des tests unitaires.

`proba_fn` est injectée dans VADSegmenter ; le chargement est paresseux et le
modèle est un singleton par processus.
"""

import threading

import numpy as np

SAMPLE_RATE = 16000

_lock = threading.Lock()
_model = None


def get_silero_probas() -> "callable":
    """Retourne proba_fn(window_f32_512) -> float, avec état Silero interne."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from silero_vad import load_silero_vad

                model = load_silero_vad(onnx=True)

                def proba(window: np.ndarray) -> float:
                    return model(torch.from_numpy(window), SAMPLE_RATE).item()

                _model = proba
    return _model
