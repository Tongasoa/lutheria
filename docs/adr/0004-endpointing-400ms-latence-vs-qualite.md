# ADR 0004 — Endpointing VAD : ~400 ms de silence, coupe à 15 s

- **Statut** : accepté
- **Date** : 2026-08-25

## Contexte

La latence bout-en-bout cible (~1,5–2,5 s) est dominée par l'attente de la fin
de phrase : le segment n'est transcrit qu'après `silence_ms` de parole
interrompue. Chaque fenêtre Silero = 512 échantillons @ 16 kHz = 32 ms.

Compromis :
- silence court (< 300 ms) → phrases coupées aux micro-pauses → traductions
  tronquées et moins naturelles ;
- silence long (> 600 ms) → traduction fidèle mais sensation de "retard".

## Décision

1. **Endpointing à 400 ms de silence** (`LUTHERIA_VAD_SILENCE_MS`, réglable) :
   compromis retenu pour prioriser la latence perçue tout en gardant des
   phrases majoritairement complètes.
2. **Pré-roll de 320 ms** (10 fenêtres) avant le déclenchement : le début de
   mot n'est pas rogné quand la détection de parole arrive 2-3 fenêtres tard.
3. **Coupe de sécurité à 15 s** (`LUTHERIA_MAX_SEGMENT_SECONDS`) : un long
   monologue est découpé en segments indépendants (limite mémoire Whisper,
   évite les hallucinations sur les segments très longs).
4. Seuil de probabilité de parole : **0,5** (défaut Silero), non exposé en v1.

## Conséquences

- (+) Latence d'endpointing constante ≈ 400 ms après le dernier mot.
- (+) Paramétrable sans modification de code (variables d'env).
- (−) Les phrases avec pauses > 400 ms sont coupées : la traduction perd du
  contexte inter-segments (assumé en v1, voir ADR 0001).
- Réglage fin possible plus tard (endpointing adaptatif) sans changer
  l'interface du segmenteur.
