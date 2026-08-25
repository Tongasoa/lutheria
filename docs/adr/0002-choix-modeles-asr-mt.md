# ADR 0002 — Choix des modèles : Whisper malgache fine-tuné + NLLB-200 distilled-600M

- **Statut** : accepté
- **Date** : 2026-08-25

## Contexte

ASR malgache → traduction française, temps réel (~1,5–2,5 s cible), serveur GPU
unique partagé entre les deux tâches. Le propriétaire du projet dispose déjà
d'un Whisper medium fine-tuné sur ses données malgaches, continuellement
amélioré : `Tongasoa/whisper-malagasy-medium-full-v2`.

## Décision

### ASR
- Modèle : **Whisper fine-tuné maison** (`LUTHERIA_ASR_MODEL`, swap sans code à
  chaque nouveau fine-tuning).
- Runtime : **faster-whisper / CTranslate2**, quantification int8 GPU (float16
  possible), `beam_size=1` (greedy) pour la latence.
- Décodage langue forcée `mg` : pas de détection de langue par segment (économie
  ~0,5 s et évite les erreurs de détection sur segments courts).
- Fallback dev CPU : même code, `device=cpu`, `compute_type=int8`.

### MT (étape 4)
- Modèle : **NLLB-200 distilled-600M** (`mlg_Latn → fra_Latn`) plutôt qu'OPUS-MT :
  couverture explicite du malgache, meilleure qualité sur langue à faibles
  ressources, taille compatible GPU partagé (~1,2 Go en int8).
- Runtime : CTranslate2 int8, beam 1-2.

## Conséquences

- (+) Le pipeline reste identique quand un meilleur fine-tuning arrive (variable
  d'env uniquement).
- (+) int8 + greedy ≈ latence ASR < 500 ms par segment de 5 s sur T4.
- (−) greedy peut sacrifier un peu de WER vs beam 4-5 ; arbitrage assumé v1,
  réglable plus tard sans changement structurel.
- (−) Erreurs en cascade ASR→MT (voir ADR 0001).
