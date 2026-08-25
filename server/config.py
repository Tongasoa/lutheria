"""Configuration centralisée via variables d'environnement (préfixe LUTHERIA_)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LUTHERIA_", env_file=".env")

    mic_token: str

    asr_model: str = "Tongasoa/whisper-malagasy-medium-full-v2"
    asr_language: str = "mg"
    asr_device: str = "cpu"  # "cuda" sur EC2 GPU
    asr_compute_type: str = "int8"  # "float16" recommandé sur GPU
    mt_model: str = "facebook/nllb-200-distilled-600M"
    mt_src_lang: str = "mlg_Latn"
    mt_tgt_lang: str = "fra_Latn"
    mt_device: str = "cpu"  # "cuda" sur EC2 GPU

    vad_silence_ms: int = 400
    max_ws_frame_bytes: int = 65536
    max_segment_seconds: int = 15
