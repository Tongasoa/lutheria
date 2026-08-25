"""MT : traduction malgache -> français via NLLB-200 sous CTranslate2 (ADR 0002).

CTranslate2 int8 : ~10-20× plus rapide que transformers/fp32 sur CPU, et
float16 sur GPU. Le modèle doit être converti au préalable :

    ./scripts/convert_ct2.sh mt

L'interface `translate(text) -> str` reste mockable ; chargement paresseux.
"""

from typing import Protocol


class MTEngine(Protocol):
    def translate(self, text: str) -> str: ...


def new_translator(model_path: str, device: str, compute_type: str):
    """Fabrique le traducteur CTranslate2 (point de monkeypatch pour les tests)."""
    import ctranslate2

    return ctranslate2.Translator(model_path, device=device, compute_type=compute_type)


def load_tokenizer(model_path: str, src_lang: str):
    """Charge le tokenizer sentencepiece HF (point de monkeypatch pour les tests)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, src_lang=src_lang)


class NLLBEngine:
    """NLLB-200 converti CTranslate2 + tokenizer sentencepiece du repo HF source.

    Le convertisseur ct2 ne copie pas le modèle sentencepiece : le tokenizer
    est chargé depuis le repo HF d'origine (petit téléchargement, mis en cache).
    """

    def __init__(
        self,
        model_path: str,
        src_lang: str = "mlg_Latn",
        tgt_lang: str = "fra_Latn",
        device: str = "cpu",
        compute_type: str = "int8",
        max_decoding_length: int = 256,
        tokenizer_name: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._tokenizer_name = tokenizer_name or model_path
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._device = device
        self._compute_type = compute_type
        self._max_decoding_length = max_decoding_length
        self._translator = None
        self._tokenizer = None

    def ensure_loaded(self) -> None:
        if self._translator is None:
            self._translator = new_translator(
                self._model_path, self._device, self._compute_type
            )
            self._tokenizer = load_tokenizer(self._tokenizer_name, self._src_lang)

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        self.ensure_loaded()
        # recette officielle CTranslate2 pour NLLB : tokens sentencepiece + prefixe langue cible
        source_tokens = self._tokenizer.convert_ids_to_tokens(
            self._tokenizer.encode(text)
        )
        results = self._translator.translate_batch(
            [source_tokens],
            target_prefix=[[self._tgt_lang]],
            max_decoding_length=self._max_decoding_length,
            beam_size=1,
        )
        output_tokens = results[0].hypotheses[0][1:]  # retirer le préfixe langue
        return self._tokenizer.decode(
            self._tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        ).strip()


def build_mt(settings) -> NLLBEngine:
    """Fabrique le moteur MT depuis la configuration (chemin modèle swap-able)."""
    return NLLBEngine(
        model_path=settings.mt_model,
        tokenizer_name=settings.mt_tokenizer_model,
        src_lang=settings.mt_src_lang,
        tgt_lang=settings.mt_tgt_lang,
        device=settings.mt_device,
        compute_type=settings.mt_compute_type,
    )
