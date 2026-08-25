# ADR 0006 — Instance EC2 g4dn.xlarge, AMI Deep Learning, arrêt à la demande

- **Statut** : accepté
- **Date** : 2026-08-25

## Contexte

Serveur unique hébergeant VAD + ASR (Whisper medium fine-tuné) + MT (NLLB-600M)
pour un flux temps réel. Contrainte : coût minimal. L'entraînement du modèle
ASR continue par ailleurs sur d'autres instances.

## Décision

| Élément | Choix | Justification |
|---|---|---|
| Type | **g4dn.xlarge** | GPU NVIDIA T4 16 Go, ~0,53 $/h on-demand — le GPU CUDA le moins cher du catalogue. VRAM ≫ besoin (~4-5 Go en int8). |
| AMI | **Deep Learning Base OSS NVIDIA Driver AMI (Ubuntu 22.04)** | Pilotes NVIDIA préinstallés et validés ; évite 30-60 min de friction driver. |
| Disque | 80 Go gp3 | OS+CUDA ~25 Go, modèles convertis ~3 Go, marge. |
| Réseau | Elastic IP **ou** DuckDNS (suit l'IP dynamique) | Let's Encrypt exige un nom de domaine. |
| Marche/arrêt | **Arrêt manuel entre les usages** ; Spot pour les longues sessions de test | 4 h/jour ≈ 63 $/mois vs ~380 $ en continu. |

Non choisis :
- `g4ad.xlarge` (GPU AMD) : incompatible CUDA/CTranslate2.
- `g5.xlarge` : 2× plus cher, A10G superflu pour 1 flux.
- CPU-only (t3…) : Whisper medium impossible en temps réel.

## Conséquences

- (+) Coût cible tenu : 30-80 $/mois on-demand, 15-30 $ en Spot.
- (+) Changement de type possible à chaud (arrêt → resize) pour réentraîner
  sur g5 puis redescendre en g4dn pour servir.
- (−) Spot = interruption possible avec 2 min de préavis → code sur GitHub +
  modèle ASR convertible rapidement (`convert_ct2.sh asr`) pour redémarrer vite.
