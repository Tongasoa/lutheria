# ADR 0001 — Architecture en cascade VAD → ASR → MT sur WebSocket

- **Statut** : accepté
- **Date** : 2026-08-25

## Contexte

Traduire en quasi-temps réel la parole malgache (malagasy) vers le français,
avec 1 client micro (navigateur) et N lecteurs (navigateurs). Serveur distant
AWS EC2. Latence cible : ~1,5–2,5 s de la fin de phrase à l'affichage français.

## Décision

Pipeline en cascade, une connexion WebSocket entrante (`/ws/mic`, PCM brut
16 kHz mono int16) alimentant :

1. **Silero VAD** (ONNX, CPU) — détection de parole et endpointing
   (silence ≥ ~400 ms ⇒ fin de segment, voir ADR 0004).
2. **ASR** — Whisper fine-tuné malgache via faster-whisper/CTranslate2 (GPU en
   prod, CPU en dev), transcription malgache.
3. **MT** — NLLB-200 distilled-600M, `mlg_Latn → fra_Latn`.

Les résultats sont diffusés (broadcast) à tous les clients de `/ws/listen`.
Deux messages par segment partagent le même `id` :

| Étape | Message | `state` |
|---|---|---|
| Fin ASR | texte malgache | `partial` |
| Fin MT | texte français (remplace/complète la ligne côté lecteur) | `final` |

Le pipeline tourne dans des tâches asyncio découplées par une `asyncio.Queue` :
le VAD n'est jamais bloqué par l'ASR/MT (pas de perte audio).

## Conséquences

- (+) Chaque brique est interchangeable (mockable en tests, remplaçable en prod).
- (+) Affichage partiel : latence perçue réduite de ~0,5 s.
- (−) Erreurs en cascade : une mauvaise transcription mg dégrade la traduction.
- (−) Pas de contexte inter-segments en v1 (chaque segment traduit isolément).
- TTS reporté en v2 : la brique MT produit déjà le texte, l'ajout sera trivial.
