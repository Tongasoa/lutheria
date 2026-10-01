#!/usr/bin/env python
"""Évalue un moteur NLLB sur Flores-200 devtest, direction plt_Latn -> fra_Latn.

Le jeu de données est celui utilisé par Meta pour les chiffres publiés de
NLLB-200 : 1012 paires alignées, texte écrit (articles Wikipédia). Il ne dit
rien de la qualité en oral malgache — il sert à *classer* deux modèles de façon
reproductible, pas à valider la perception des lecteurs.

    python scripts/eval_mt.py --model models/mt-nllb \
        --tokenizer facebook/nllb-200-distilled-600M --limit 300

`--limit 200` suffit pour classer deux modèles ; 1012 (défaut) pour le chiffre
définitif. Le décodage est celui de la production (beam 1, `server/mt.py`),
sinon on mesurerait autre chose que ce qui tourne en prod.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
CACHE = Path.home() / ".cache" / "lutheria" / "flores200"

# Le harnais mesure plt_Latn -> fra_Latn. Ces codes nomment les fichiers du jeu,
# pas le préfixe donné au modèle : ce sont deux choses, et les confondre fait
# chercher un fichier `mlg_Latn.devtest` qui n'existe pas.
FLORES_SRC = "plt_Latn"
FLORES_TGT = "fra_Latn"


def ensure_flores() -> Path:
    """Télécharge et extrait Flores-200 dans le cache local (source Meta, sans gate)."""
    root = CACHE / "flores200_dataset"
    if (root / "devtest").is_dir():
        return root
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "flores200_dataset.tar.gz"
    if not archive.exists():
        print(f"téléchargement de Flores-200 depuis {FLORES_URL} ...")
        with urllib.request.urlopen(FLORES_URL) as resp, archive.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    with tarfile.open(archive) as tar:
        tar.extractall(CACHE, filter="data")
    return root


def load_pairs(root: Path, split: str, limit: int | None) -> tuple[list[str], list[str]]:
    """Charge les phrases source et référence ; l'alignement FLORES est indexé par ligne."""
    def read(code: str) -> list[str]:
        path = root / split / f"{code}.{split}"
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    srcs, refs = read(FLORES_SRC), read(FLORES_TGT)
    if len(srcs) != len(refs):
        raise SystemExit(f"désalignement FLORES : {len(srcs)} source vs {len(refs)} référence")
    n = limit or len(srcs)
    return srcs[:n], refs[:n]


def check_lang_code(tokenizer, code: str, role: str, allow_unk: bool = False) -> int:
    """Refuse un code de langue que le tokenizer ne résout pas.

    NllbTokenizer ne lève rien sur un code hors FLORES-200 : `convert_tokens_to_ids`
    renvoie l'id de <unk>, qui part ensuite en préfixe. Le modèle traduit quand
    même, si bien que la faute est invisible — c'était le bug du préfixe <unk>
    (ADR 0008). L'outil qui sert à mesurer la qualité doit donc être strict.

    `allow_unk` existe pour quantifier ce bug rétrospectivement (cellule A du
    tableau A/B/C), jamais par inadvertance.
    """
    token_id = tokenizer.convert_tokens_to_ids(code)
    if token_id == tokenizer.unk_token_id:
        message = (
            f"code langue {role}={code!r} absent du vocabulaire : il résoudrait vers "
            f"<unk> (id {token_id}). Codes FLORES-200 valides pour le malgache : plt_Latn."
        )
        if not allow_unk:
            raise SystemExit(message)
        print(f"ATTENTION — mesure d'une configuration fautive : {message}")
    return token_id


def evaluate(engine, srcs: list[str], refs: list[str], warmup: int = 1
             ) -> tuple[dict, list[str]]:
    """Traduit, puis agrège qualité (sacrebleu) et latence par phrase."""
    import sacrebleu

    for text in srcs[:warmup]:  # amorçage : le 1er appel paie le chargement CUDA/CT2
        engine.translate(text)

    hyps: list[str] = []
    latencies: list[float] = []
    for i, src in enumerate(srcs, 1):
        start = time.perf_counter()
        hyps.append(engine.translate(src))
        latencies.append(time.perf_counter() - start)
        if i % 100 == 0:
            print(f"  {i}/{len(srcs)} …")

    ms = np.array(latencies) * 1000
    return {
        "n": len(srcs),
        "chrf++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 3),
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 3),
        "latence_ms_p50": round(float(np.percentile(ms, 50)), 1),
        "latence_ms_p95": round(float(np.percentile(ms, 95)), 1),
        "latence_ms_moy": round(statistics.fmean(ms), 1),
    }, hyps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="répertoire converti CTranslate2")
    p.add_argument("--tokenizer", required=True, help="repo HF du tokenizer NLLB")
    p.add_argument("--src-lang", default="plt_Latn")
    p.add_argument("--tgt-lang", default="fra_Latn")
    p.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    p.add_argument("--limit", type=int, default=None, help="sous-ensemble de phrases")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--label", default=None, help="nom de la cellule dans le tableau A/B/C")
    p.add_argument("--allow-unk-src", action="store_true",
                   help="diagnostic uniquement : autorise un code source hors FLORES-200")
    p.add_argument("--hypotheses", default=None, help="fichier .txt de sortie pour comparaison")
    p.add_argument("--json", default=None, help="fichier .json de métriques")
    args = p.parse_args()

    from server.mt import load_tokenizer
    from server.mt import NLLBEngine

    tokenizer = load_tokenizer(args.tokenizer, args.src_lang)
    check_lang_code(tokenizer, args.src_lang, "source", allow_unk=args.allow_unk_src)
    check_lang_code(tokenizer, args.tgt_lang, "cible")

    if args.src_lang != FLORES_SRC or args.tgt_lang != FLORES_TGT:
        print(f"ATTENTION — les références restent {FLORES_SRC} -> {FLORES_TGT} alors que le "
              f"modèle reçoit {args.src_lang} -> {args.tgt_lang} : le score ne sera pas "
              f"interprétable.")

    root = ensure_flores()
    srcs, refs = load_pairs(root, args.split, args.limit)
    print(f"Flores-200 {args.split} : {len(srcs)} paires "
          f"{args.src_lang} -> {args.tgt_lang} ({args.device}/{args.compute_type})")

    engine = NLLBEngine(
        model_path=args.model,
        tokenizer_name=args.tokenizer,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        device=args.device,
        compute_type=args.compute_type,
    )
    metrics, hyps = evaluate(engine, srcs, refs)
    metrics["label"] = args.label or args.model
    metrics["src_lang"] = args.src_lang

    width = max(len(k) for k in metrics)
    print(f"\n=== {metrics['label']} ===")
    for key, value in metrics.items():
        print(f"  {key:<{width}} : {value}")

    if args.hypotheses:
        Path(args.hypotheses).write_text("\n".join(hyps), encoding="utf-8")
        print(f"hypothèses écrites dans {args.hypotheses}")
    if args.json:
        Path(args.json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"métriques écrites dans {args.json}")


if __name__ == "__main__":
    main()