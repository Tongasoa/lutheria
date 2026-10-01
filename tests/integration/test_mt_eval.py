"""Intégration du harnais d'évaluation MT (marque `integration`).

Deux garde-fous sur `scripts/eval_mt.py` : le refus d'un code de langue hors
FLORES-200, et le fait que l'évaluation produit réellement des métriques sur le
vrai moteur. Le premier protège le bug du préfixe <unk> (ADR 0008) ; le second
évite que le harnais casse en silence au fil des évolutions de `server/mt.py`.

Pré-requis : `convert_ct2.sh mt` et `sacrebleu` (requirements.txt).
"""

import pytest

from scripts.eval_mt import check_lang_code, ensure_flores, evaluate, load_pairs
from server.mt import NLLBEngine, load_tokenizer

TOKENIZER = "facebook/nllb-200-distilled-600M"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(TOKENIZER, "plt_Latn")


@pytest.mark.integration
def test_garde_fou_refuse_un_code_hors_flores(tokenizer):
    check_lang_code(tokenizer, "plt_Latn", "source")  # code valide : ne lève rien
    with pytest.raises(SystemExit):
        check_lang_code(tokenizer, "mlg_Latn", "source")


@pytest.mark.integration
def test_flores_devtest_plt_et_fra_sont_alignes():
    srcs, refs = load_pairs(ensure_flores(), "devtest", 5)
    assert len(srcs) == len(refs) == 5
    assert all(s and r for s, r in zip(srcs, refs))


@pytest.mark.integration
def test_le_harnais_produit_des_metriques_sur_le_vrai_moteur():
    srcs, refs = load_pairs(ensure_flores(), "devtest", 5)
    engine = NLLBEngine(
        model_path="models/mt-nllb",
        tokenizer_name=TOKENIZER,
        src_lang="plt_Latn",
        tgt_lang="fra_Latn",
        device="cpu",
        compute_type="int8",
    )
    metrics, hyps = evaluate(engine, srcs, refs)
    assert metrics["n"] == 5 and len(hyps) == 5
    assert 0.0 < metrics["chrf++"] < 100.0
    assert metrics["latence_ms_p50"] > 0.0