"""MT : traduction malgache -> français via NLLB-200 (voir ADR 0002).

Interface minimale `translate(text) -> str` pour rester mockable ; le modèle
est chargé paresseusement à la première utilisation.
"""

from typing import Protocol


class MTEngine(Protocol):
    def translate(self, text: str) -> str: ...


def _load_model_and_tokenizer(model_name: str, device: str):
    """Point d'injection pour les tests — charge le vrai modèle transformers."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device).eval()
    if device.startswith("cuda"):
        model = model.half()
    return model, tokenizer


class NLLBEngine:
    """NLLB-200 : texte source (mlg) -> texte cible (fra), beam court pour la latence."""

    def __init__(
        self,
        model_name: str,
        src_lang: str = "mlg_Latn",
        tgt_lang: str = "fra_Latn",
        device: str = "cpu",
        max_new_tokens: int = 256,
    ) -> None:
        self._model_name = model_name
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._tokenizer = _load_model_and_tokenizer(
                self._model_name, self._device
            )

    def translate(self, text: str) -> str:
        self.ensure_loaded()
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        generated = self._model.generate(
            **inputs,
            forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(self._tgt_lang),
            max_new_tokens=self._max_new_tokens,
            num_beams=1,
        )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def build_mt(settings) -> NLLBEngine:
    """Fabrique le moteur MT depuis la configuration."""
    return NLLBEngine(
        model_name=settings.mt_model,
        src_lang=settings.mt_src_lang,
        tgt_lang=settings.mt_tgt_lang,
        device=settings.mt_device,
    )
