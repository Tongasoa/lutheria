# ADR 0003 — PCM brut en WebSocket, compression Opus reportée

- **Statut** : accepté
- **Date** : 2026-08-25

## Contexte

Le micro du navigateur envoie l'audio au serveur EC2 distant. Deux options :
PCM int16 mono 16 kHz brut (32 Ko/s) ou compression Opus (~6-24 Ko/s selon le
bitrate). Le débit montant des connexions modernes (ADSL ≥ 1 Mb/s, 4G ≥ 5 Mb/s)
absorbe largement 32 Ko/s.

## Décision

1. **v1 : PCM brut**, frames binaires WS de ~100 ms (3200 octets).
2. **Opus reporté en v2**, déclencheurs : latence réseau mesurée anormale,
   utilisateurs sur liens très contraints, ou facturation bande passante
   significative.

## Conséquences

- (+) Zéro encodage/décodage : moins de code, moins de CPU client, aucun risque
  de perte qualité liée au codec avant ASR.
- (+) Le serveur reçoit exactement ce que Whisper attend après normalisation.
- (−) 32 Ko/s par émetteur — sans objet en v1 (un seul producteur).
- Migration facile : le protocole WS est déjà binaire ; seul le pipeline de
  décodage côté serveur change (ajout d'un décodeur Opus avant le VAD).
