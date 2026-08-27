# Lutheria

Traduction vocale en temps réel **malgache → français** : un émetteur parle,
plusieurs lecteurs suivent la traduction à l'écrit avec ~1,5–2,5 s de délai.

```
Client micro (navigateur)                 Lecteurs ×N (navigateurs)
  PCM 16 kHz ──WSS──▶ ┌──────────────────────────┐ ──WSS──▶ listen.html
                      │ FastAPI                  │
                      │  Silero VAD (endpointing │
                      │   400 ms, coupe 15 s)    │
                      │  Whisper mg fine-tuné    │
                      │   (faster-whisper/CT2)   │
                      │  NLLB-200 mlg→fra (CT2)  │
                      └──────────────────────────┘
```

Chaque segment produit deux messages : `partial` (texte malgache affiché
immédiatement) puis `final` (traduction française qui complète la ligne,
même `id`). Détails des choix : [docs/adr/](docs/adr/).

## Structure

| Répertoire | Contenu |
|---|---|
| `server/` | API WebSocket (`main.py`), pipeline (`vad.py`, `asr.py`, `mt.py`), config |
| `client/` | pages web émetteur + lecteurs (vanilla JS, zéro build) |
| `tests/unit/` | logique testée sans modèles (TDD, <2 s) |
| `tests/integration/` | vrais modèles (marque `integration`) |
| `docs/adr/` | décisions architecturales datées |
| `scripts/` | conversion CTranslate2, certificat, client de fumée |
| `deploy/` | nginx, systemd, [guide EC2 pas à pas](deploy/DEPLOYMENT.md) |

## Démarrage rapide (dev local, Mac OK)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt  # 3.11–3.13 OK (prod 26.04 en 3.13, voir ADR 0007)
cp .env.example .env                    # renseigner LUTHERIA_MIC_TOKEN
./scripts/convert_ct2.sh asr            # une fois (~3 Go téléchargés)
./scripts/convert_ct2.sh mt             # une fois (~2,5 Go)
.venv/bin/uvicorn server.main:create_app --factory --port 8000
# → http://127.0.0.1:8000  (mic.html = émettre, listen.html = écouter)
```

Sur Mac sans GPU : tout tourne en CPU int8 — suffisant pour développer ; le
temps réel exige la T4 (déploiement, voir `deploy/DEPLOYMENT.md` + ADR 0007).

## Tests

```bash
.venv/bin/pytest            # unitaires uniquement (<2 s), modèles mockés
.venv/bin/pytest -m integration   # vrais modèles (téléchargement au premier run)
```

## Déploiement

Voir [deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md) : instance g4dn.xlarge,
conversion des modèles, systemd, nginx TLS, validation par
`scripts/smoke_client.py`.

## Conventions

- TDD : tests d'abord, modèles ML toujours injectables/mockables.
- Commits atomiques ~100 lignes, Conventional Commits.
- Toute décision structurante → un ADR dans `docs/adr/`.
- Secrets uniquement via variables d'environnement (jamais commités).
- Les agents codants lisent [AGENTS.md](AGENTS.md).
