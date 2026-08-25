"""Tests du moteur de traduction NLLB (modèle interne simulé)."""

import numpy as np
import pytest

from server.mt import NLLBEngine


class FakeInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors=None):
        self.calls.append(text)
        return FakeInputs(input_ids=[[1, 2, 3]])

    def convert_tokens_to_ids(self, lang):
        return hash(lang) % 1000

    def batch_decode(self, tokens, skip_special_tokens=True):
        return ["  Bonjour comment allez-vous ?  "]


class FakeModel:
    device = "cpu"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_with = None

    def generate(self, **kwargs):
        self.generated_with = kwargs
        return [[9, 9, 9]]


def make_engine():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    engine = NLLBEngine(
        model_name="fake", src_lang="mlg_Latn", tgt_lang="fra_Latn", device="cpu"
    )
    engine._tokenizer = tokenizer
    engine._model = model
    return engine, tokenizer, model


class TestNLLBEngine:
    def test_translate_nettoie_les_espaces(self):
        engine, _, _ = make_engine()
        assert engine.translate("manao ahoana") == "Bonjour comment allez-vous ?"

    def test_langue_source_passee_au_tokenizer(self):
        engine, tokenizer, _ = make_engine()
        engine.translate("salama")
        assert tokenizer.calls == ["salama"]

    def test_langue_cible_forcee_via_bos(self):
        engine, _, model = make_engine()
        engine.translate("salama")
        assert "forced_bos_token_id" in model.generated_with

    def test_chargement_paresseux_reel(self, monkeypatch):
        """Sans modèle injecté, ensure_loaded passe par transformers."""
        engine = NLLBEngine(model_name="fake", device="cpu")
        called = {}

        def fake_from_pretrained(name, device="cpu"):
            called["model"] = name
            return FakeModel(FakeTokenizer()), FakeTokenizer()

        import server.mt as mt_module
        monkeypatch.setattr(mt_module, "_load_model_and_tokenizer", fake_from_pretrained)
        engine.ensure_loaded()
        assert called["model"] == "fake"
