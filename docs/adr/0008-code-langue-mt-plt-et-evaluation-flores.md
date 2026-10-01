# ADR 0008 — Code langue MT `plt_Latn`, harnais d'évaluation Flores-200, rejet de `francis47/nllb_mg_v3_3ep`

- **Statut** : accepté
- **Date** : 2026-10-01
- **Amende** : ADR 0002 (MT — choix des modèles)
- **Réf** : `server/config.py:17`, `server/mt.py:25-45`, `tests/unit/test_config.py:26`, `tests/unit/test_mt.py:97-125`, `scripts/eval_mt.py`, `docs/adr/0001:21`

## Contexte

Deux défauts sont apparus dans la même semaine, tous deux **silencieux** :

1. Le code langue source MT était `mlg_Latn`. `mlg` est le code **ISO 639-3**
   du malgache, mais NLLB utilise les codes **FLORES-200**, où le malgache vaut
   **`plt_Latn`** (Plateau Malagasy). Hors FLORES-200, `NllbTokenizer` ne lève
   rien : `convert_tokens_to_ids` rend l'id de `<unk>` (3), qui partait en
   préfixe source. Le modèle déduisait la langue du texte, d'où une traduction
   plausible, aucun crash, aucune trace dans les logs — et un service
   `active (running)` pendant tout ce temps.
2. Aucun harnais de qualité n'existait : le test MT d'intégration ne vérifiait
   que `len(traduction) > 2`. Impossible de démontrer un gain ou une régression.

S'y ajoutait une proposition de bascule vers un fine-tune Malagasy
(`francis47/nllb_mg_v3_3ep`) afin d'améliorer la traduction.

Le même ítem est à l'origine de la campagne de durcissement qui accompagne cet
ADR : CI minimale Python 3.13, tests unitaires réellement sans modèle, garde-fous
explicites.

## Décision

### 1. Code langue source : `plt_Latn`

`mlg_Latn` est remplacé par `plt_Latn` partout (config, moteur, `.env.example`,
ADR 0001/0002). Garde-fou `test_lang_codes_mt_sont_valides` vérifiant
l'appartenance à `FAIRSEQ_LANGUAGE_CODES` — hors liste, test rouge.

### 2. `load_tokenizer` refuse un code non résolu au lieu de le deviner

Les codes de langue NLLB vivent dans le `tokenizer.json` (champ `added_tokens`)
que publie Meta, **pas** dans le `sentencepiece.bpe.model` : sur le modèle
officiel, le SPM fait 256 000 pièces et n'en contient aucune.

Les fine-tunes communautaires omettent souvent ce `tokenizer.json`. On pourrait
réinjecter `FAIRSEQ_LANGUAGE_CODES` pour peupler le vocabulaire — c'est
**explicitement écarté** : un identifiant reconstruit à la bonne place dans la
plage libre n'est pas forcément celui de l'entraînement, et le modèle produit
alors un texte plausible et faux. Le chargement échoue donc bruyamment.

### 3. Harnais d'évaluation : Flores-200 devtest

`scripts/eval_mt.py` traduit les 1012 paires `plt_Latn → fra_Latn` de
**devtest** via `NLLBEngine`, avec le décodage de la production (beam 1), puis
rapporte chrF++, BLEU et latence p50/p95. Jeu obtenu depuis
`dl.fbaipublicfiles.com` (source Meta, sans gate) — le portage Hugging Face
`openlanguagedata/flores_plus` est gated et exige un token à accès aux dépôts
gated publics.

Règle de décision, **relative** et fixée avant mesure :
`déployer C si chrF++(C) > chrF++(B) et p95(C) ≤ 1,25 × p95(B)`.

### 4. `francis47/nllb_mg_v1` et `nllb_mg_v3_3ep` : rejetés

Résultats sur 300 paires de devtest, CPU int8, latence par phrase :

| Cellule | src_lang | chrF++ | BLEU | p50 (ms) | p95 (ms) | minuscule initiale |
|---|---|---|---|---|---|---|
| A — 600M avant correctif | `mlg_Latn` | 41,962 | 18,877 | 793 | 1 679 | — |
| B — 600M prod actuelle | `plt_Latn` | **48,675** | **25,747** | 630 | 1 090 | 0 % |
| C — `nllb_mg_v1` | `plt_Latn` | 44,713 | 18,647 | 668 | 1 126 | 95 % |
| D — `nllb_mg_v3_3ep` | `plt_Latn` | 31,589 | 8,408 | 650 | 3 363 | 93 % |

**`nllb_mg_v1`** est techniquement sain, contrairement à la v3 : `vocab_size`
256 206 identique à l'officiel, base déclarée
(`facebook/nllb-200-distilled-600M`), et un `tokenizer.json` dont le vocabulaire
et la segmentation sont **identiques** au modèle Meta (`plt_Latn` = 256 119,
`fra_Latn` = 256 057). Rien à deviner, comparaison propre. Il perd malgré tout
la règle sur la qualité : **−3,96 chrF++** et **−7,1 BLEU**. La latence passe
(x1,03), mais ce n'est pas le critère limitant.

Il produit par ailleurs une **minuscule initiale dans 95 % des cas** (officiel :
0 %) — défaut visible par les lecteurs. Ce comportement, cumulé à la baisse sur
Flores, est cohérent avec un corpus d'entraînement en minuscules et de nature
conversationnelle : Flores mesure de l'écrit de type Wikipédia, et le métrique
est alors probablement le mauvais juge. Lever ce doute demanderait des
références en langue-domain, écartées ici.

**`nllb_mg_v3_3ep`** échoue plus nettement et pour une raison technique : le dépôt
ne publie pas de `tokenizer.json`, donc `plt_Latn` **et** `fra_Latn` résolvent
vers `<unk>` — le modèle ne reçoit ni sa langue source ni l'ordre de produire du
français. La reconstruction des identifiants a été tentée (`plt_Latn`=268205,
`fra_Latn`=268143 pour un `vocab_size` de 268291, donc dans la plage libre de
l'embedding) et le moteur tourne, mais la sortie est du texte plausible et faux
(« ne vous inquiétez pas d'aller à l'anxiété »). S'y ajoutent l'absence de
licence déclarée, de model card, de métriques et de base de fine-tuning
documentée.

Les deux dépôts sans model card ni licence déclarée restent exclusifs de la
prod : la base NLLB-200 est CC-BY-NC et « non destiné au déploiement en
production ».

### 5. NLLB-200 distilled-600M conservé

## Conséquences

- (+) Le service diffuse désormais du français correct : **+6,7 chrF++** et
  **+6,9 BLEU** pour une modification d'une ligne, sans coût de latence.
- (+) Les fautes de cette famille sont désormais visibles : garde-fou de config,
  refus bruyant au chargement du tokenizer, refus explicite dans `eval_mt.py`.
- (+) Toute évolution du MT se mesure avant déploiement, avec une règle écrite
  d'avance ; plus de décision fondée sur « ça a l'air bien ».
- (−) Flores-200 mesure du **texte écrit** (articles Wikipédia) : il classe des
  modèles, il ne prédit pas la qualité en oral malgache. C'est probablement la
  raison pour laquelle `nllb_mg_v1`, vraisemblablement entraîné sur de l'oral en
  minuscules, sort en dessous. La perception des lecteurs reste à valider au
  navigateur, et aucun jeu en langue-domain n'existe pour trancher.
- (−) Un dépôt tiers sans `tokenizer.json` est désormais rejeté : c'est
  délibéré, mais cela ferme l'accès à la plupart des fine-tunes communautaires.
  Les modèles Meta restent interchangeables par variable d'environnement.
- (−) La CI n'exerce pas les imports paresseux : une faute de frappe dans une
  fabrique n'apparaît qu'en intégration, sur la machine de dev ou la prod.
- Suite possible, non décidée : `facebook/nllb-200-distilled-1.3B` (tokenizer
  vérifié sain, provenance Meta) pour ~2× la latence MT, à confronter à la même
  règle ×1,25.