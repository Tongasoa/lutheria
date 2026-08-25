# AGENTS.md — projet lutheria

> **Scope :** ce fichier configure les agents travaillant sur le dépôt
> `lutheria` (traduction vocale malgache→français). Les compétences
> réutilisables sont dans `skills/`, hors périmètre applicatif.

## Vue d'ensemble

Application client-serveur de traduction vocale temps réel :

- **Serveur** : FastAPI + pipeline asyncio `VAD (Silero) → ASR (Whisper mg,
  faster-whisper/CTranslate2) → MT (NLLB-200, CTranslate2)` ; deux endpoints WS
  (`/ws/mic` producteur authentifié par token, `/ws/listen` lecteurs en
  broadcast). Le worker du pipeline vit dans le **lifespan** de l'app.
- **Clients** : pages vanilla JS servies par FastAPI (`client/`), capture PCM
  via AudioWorklet, affichage avec patch partial→final.
- **Décisions structurantes** : lire `docs/adr/0001…0006` avant toute
  modification d'architecture. Toute nouvelle décision = nouvel ADR.

## Commandes

```bash
source .venv/bin/activate            # Python 3.11
pytest                               # unitaires uniquement (<2 s) — OBLIGATOIRE vert avant commit
pytest -m integration                # vrais modèles (lents, téléchargements au premier run)
uvicorn server.main:create_app --factory --port 8000   # lancer le serveur en local
./scripts/convert_ct2.sh asr|mt      # (re)convertir les modèles CTranslate2
python scripts/smoke_client.py --server ws://127.0.0.1:8000 --token X --file f.wav --insecure
```

## Conventions impératives

1. **TDD** : tests écrits avant le code ; les modèles ML sont toujours
   injectés via des fabriques (`vad_factory`, `asr_factory`, `mt_factory`) et
   simulés dans les tests unitaires. Aucun test unitaire ne doit charger un
   modèle réel.
2. **Git** : commits atomiques (~100 lignes), Conventional Commits (`feat:`,
   `fix:`, `docs:`, `test:`, `refactor:`, `chore:`). Ne jamais commiter :
   `.env`, `models/`, `deploy/certs/`, tout secret.
3. **Sécurité** : validation des entrées WS côté serveur (tailles de frames,
   unicité du producteur) ; secrets uniquement par variables d'environnement ;
   comparaison de token en temps constant (`hmac.compare_digest`).
4. **Protocole WS figé** : messages JSON `{id, ts, state: "partial"|"final",
   text_mg[, text_fr]}` — un même `id` pour les deux phases d'un segment.
   Tout changement → ADR + version.
5. **Style** : pas de commentaires superflus ; docstrings en français courtes
   expliquant le *pourquoi* ; typage sur les signatures publiques.

## Pièges connus

- Chaque session WebSocket TestClient a son propre event loop : utiliser
  `with TestClient(app)` (lifespan) sinon le worker n'existe pas — voir
  `tests/unit/test_ws_pipeline.py`.
- Le convertisseur CT2 ne copie pas le tokenizer NLLB : il est chargé depuis
  le repo HF source (`LUTHERIA_MT_TOKENIZER_MODEL`).
- Silero exige des fenêtres de 512 échantillons @16 kHz exactement ; passer par
  `VADSegmenter.process()` qui bufferise les tailles arbitraires.
