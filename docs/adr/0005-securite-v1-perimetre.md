# ADR 0005 — Périmètre de sécurité v1

- **Statut** : accepté
- **Date** : 2026-08-25

## Contexte

v1 mono-producteur (1 micro) / multi-lecteurs, déployée sur EC2 sans nom de
domaine (certificat TLS auto-signé). Menaces réalistes : tiers envoyant de
l'audio (coût GPU, spam sur les lecteurs), fuzzing des WebSockets, fuite de
secrets dans git.

## Décision

1. **Authentification du producteur** : `/ws/mic` exige un token
   (`LUTHERIA_MIC_TOKEN`) passé en query string au handshake ; refus (code 4401)
   avant `accept()` si manquant/invalide. Un seul producteur actif : la seconde
   connexion est refusée (code 4409).
2. **Lecteurs ouverts** (`/ws/listen`, lecture seule, aucune donnée sensible).
3. **Validation des entrées WS** : frames binaires ≤ `LUTHERIA_MAX_WS_FRAME_BYTES`
   (64 Ko), segments audio ≤ `LUTHERIA_MAX_SEGMENT_SECONDS` (15 s, coupe à
   blanc), types de messages inattendus ignorés.
4. **Secrets** : uniquement via variables d'environnement (`pydantic-settings`,
   fichier `.env` non commité, `EnvironmentFile=` systemd en prod droits 600).
5. **Réseau** : uvicorn bindé sur `127.0.0.1:8000`, seul Nginx exposé (443, TLS
   même auto-signé, WSS jamais en clair) ; SSH restreint par IP.

## Conséquences

- (+) Auth et validation présentes dès le premier commit (structurel, pas de
  migration ultérieure).
- (−) Non couvert en v1 (assumé, à revisiter avant ouverture publique) : auth
  des lecteurs, rotation des tokens, rate limiting applicatif fin, audit log.
- Migration vers HTTPS valide (DuckDNS/Let's Encrypt, puis vrai domaine) prévue
  et indolore : seule la config Nginx change.
