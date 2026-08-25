"""Tests d'intégration MT avec le vrai NLLB-200 converti CTranslate2 (marque `integration`).

Pré-requis : ./scripts/convert_ct2.sh mt  (télécharge ~2,5 Go une seule fois).
"""

import pytest

from server.config import Settings
from server.mt import build_mt


@pytest.fixture(scope="module")
def engine():
    return build_mt(Settings(mic_token="x", _env_file=None))


@pytest.mark.integration
def test_traduction_mg_vers_fr(engine):
    # phrase simple en malgache
    text = engine.translate("Manao ahoana ianao?")
    assert isinstance(text, str)
    assert len(text) > 2


@pytest.mark.integration
def test_texte_vide_renvoie_chaine_vide_sans_charger_le_modele(engine):
    text = engine.translate("   ")
    assert text == ""


@pytest.mark.integration
def test_phrase_longue_termine_en_temps_raisonnable(engine):
    long_text = "Misaotra anao nizara ny sary an-tsaina tamin'ny fivoriana teo aloha. " * 10
    text = engine.translate(long_text)
    assert isinstance(text, str) and len(text) > 0
