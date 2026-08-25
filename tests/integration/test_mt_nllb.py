"""Tests d'intégration MT avec le vrai NLLB-200 distilled-600M (marque `integration`).

Premier lancement : téléchargement ~2,5 Go dans le cache HF.
"""

import pytest

from server.mt import NLLBEngine


@pytest.fixture(scope="module")
def engine():
    return NLLBEngine(model_name="facebook/nllb-200-distilled-600M", device="cpu")


@pytest.mark.integration
def test_traduction_mg_vers_fr(engine):
    # phrase simple en malgache
    text = engine.translate("Manao ahoana ianao?")
    assert isinstance(text, str)
    assert len(text) > 2


@pytest.mark.integration
def test_texte_vide_ne_plante_pas(engine):
    text = engine.translate(" ")
    assert isinstance(text, str)


@pytest.mark.integration
def test_phrase_longue_limitee_par_max_new_tokens(engine):
    long_text = "Misaotra anao nizara ny sary an-tsaina tamin'ny fivoriana teo aloha. " * 10
    text = engine.translate(long_text)
    assert isinstance(text, str) and len(text) > 0
