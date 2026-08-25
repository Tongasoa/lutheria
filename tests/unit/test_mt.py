"""Tests du moteur de traduction NLLB/CTranslate2 (traducteur interne simulé)."""

import pytest

from server.mt import NLLBEngine


class FakeTokenizer:
    def __init__(self):
        self.encoded = None
        self.decoded = None

    def encode(self, text):
        self.encoded = text
        return [10, 20, 30]

    def convert_ids_to_tokens(self, ids):
        return [f"tok{i}" for i in ids]

    def convert_tokens_to_ids(self, tokens):
        return [int(t[3:]) if t.startswith("tok") else 0 for t in tokens]

    def decode(self, ids, skip_special_tokens=True):
        self.decoded = ids
        return "  Bonjour comment allez-vous ?  "


class FakeTranslator:
    def __init__(self):
        self.kwargs = None

    def translate_batch(self, batch, **kwargs):
        self.kwargs = kwargs
        self.batch = batch
        return [type("R", (), {"hypotheses": [["fra_Latn", "bon", "jour"]]})()]


def make_engine():
    tok, tr = FakeTokenizer(), FakeTranslator()
    engine = NLLBEngine(model_path="fake", device="cpu")
    engine._tokenizer = tok
    engine._translator = tr
    return engine, tok, tr


def test_translate_nettoie_les_espaces_et_le_prefixe_langue():
    engine, _, _ = make_engine()
    assert engine.translate("manao ahoana") == "Bonjour comment allez-vous ?"


def test_texte_vide_renvoie_chaine_vide_sans_appeler_le_modele():
    engine, tok, _ = make_engine()
    assert engine.translate("   ") == ""
    assert tok.encoded is None  # tokenizer jamais sollicité


def test_prefixe_langue_cible_et_beam_un():
    engine, _, tr = make_engine()
    engine.translate("salama")
    assert tr.kwargs["target_prefix"] == [["fra_Latn"]]
    assert tr.kwargs["beam_size"] == 1
    assert tr.batch == [["tok10", "tok20", "tok30"]]


def test_chargement_paresseux(monkeypatch):
    engine = NLLBEngine(model_path="fake-model", device="cpu", compute_type="int8")
    calls = {}

    def fake_translator(path, device, compute_type):
        calls.update(path=path, device=device, compute_type=compute_type)
        return FakeTranslator()

    import server.mt as mt_module
    monkeypatch.setattr(mt_module, "new_translator", fake_translator)
    monkeypatch.setattr(mt_module, "load_tokenizer", lambda p, s: FakeTokenizer())
    engine.ensure_loaded()
    assert calls == {"path": "fake-model", "device": "cpu", "compute_type": "int8"}
