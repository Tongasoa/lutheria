# ADR 0007 — Ajustements prod : Ubuntu 26.04, Python 3.13, dépendances CUDA, réseau et outillage

- **Statut** : accepté
- **Date** : 2026-08-28
- **Amende** : ADR 0006 (partiellement)
- **Réf** : ADR 0005 (sécurité v1), `deploy/DEPLOYMENT.md`, `server/config.py:6`, `deploy/lutheria.service:23`, `deploy/nginx-lutheria.conf:15`, `scripts/smoke_client.py:52`

## Contexte

Déploiement prod du 27–28 août 2026 sur `g4dn.xlarge` :

- L'AMI de référence `Deep Learning Base OSS NVIDIA Driver (Ubuntu 22.04)` (`ADR 0006:17`, `deploy/DEPLOYMENT.md:3`) n'est plus proposée sur la console 2026 ; l'instance lancée est **Ubuntu 26.04 LTS resolute** (AMI `Deep Learning Base OSS NVIDIA Driver` 26.04, `lsb_release 26.04`).
- Python système `3.13`/`3.14` présents (`/usr/bin/python3.13`, `deadsnakes/ppa` déjà activé) ; `python3.11-venv` indisponible (`Candidate: (none)` `apt-cache policy`), `python3.13-venv` installé `3.13.15-1+resolute1`.
- `faster-whisper`/`ctranslate2` liés CUDA 12 attendent `libcublas.so.12` ; l'AMI 26.04 ne fournit que `libcublas.so.13` (`/opt/pytorch/cuda/lib`) → `RuntimeError: Library libcublas.so.12 is not found` `server/asr.py:49` `journalctl 21:48:30`, pipeline `server/main.py:84` muet.
- `EnvironmentFile=/etc/lutheria.env` `deploy/lutheria.service:23` n'ignore pas les commentaires inline (`#`) contrairement à `pydantic-settings` `server/config.py:6` : `LUTHERIA_ASR_MODEL=models/asr-mg  # commentaire` devient valeur littérale → `HFValidationError: Repo id must be ...` `server/asr.py:39` `journalctl 21:23:58`.
- `Security Group` documenté `22`+`443` (`deploy/DEPLOYMENT.md:14`) ferme `80` → `certbot --nginx` `http-01` timeout `Fetching http://lutheria.duckdns.org/.well-known/acme-challenge ...` + `nginx` `server_name _;` `deploy/nginx-lutheria.conf:15,22` → `Could not find matching server block` et `unknown directive "lutheria.duckdns.org"` après édition malformée.
- `scripts/smoke_client.py:52` passe `ssl=None` explicitement sur `wss://` → `websockets 17.0.1` lève `ValueError: ssl=None is incompatible with a wss:// URI` (`asyncio/client.py:402`), alors que `ws://` tolérait `None`.

## Décision

1. **AMI** : cible `Deep Learning Base OSS NVIDIA Driver AMI (Ubuntu 26.04)` (fallback `24.04` si 26.04 indisponible). Disque `80 Go gp3` inchangé. Réf à mettre à jour dans `deploy/DEPLOYMENT.md:3`.
2. **Python** : système `3.13` (ou `3.x` disponible), venv `python3 -m venv .venv` ou `python3.13 -m venv`. Code compatible `3.11–3.13` (vérifié `requirements.txt:1`, CI `pytest` <2 s).
3. **Dépendances CUDA** : après `apt update`, installer `sudo apt install -y libcublas12 libcublaslt12` (sur 26.04 `resolute` + repo `developer.download.nvidia.com/compute/cuda/repos/ubuntu2604`). Alternative méta `nvidia-cuda-toolkit`. Vérif `ldconfig -p | grep cublas` doit montrer `.12` avant `systemctl restart lutheria`.
4. **Réseau** : SG entrées `22` (restreint à ton IP) + `80` (`0.0.0.0/0`, requis redirect `deploy/nginx-lutheria.conf:13` + challenge Let's Encrypt) + `443`. `8000` reste fermé (`lutheria.service:27` `127.0.0.1:8000`).
5. **Secrets** : `EnvironmentFile` strict — pas de `#` inline, pas de `$(...)`, pas de guillemets. Générer hors fichier puis `printf` :
   ```bash
   TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
   sudo install -m 600 /dev/null /etc/lutheria.env
   printf "LUTHERIA_MIC_TOKEN=%s\nLUTHERIA_ASR_DEVICE=cuda\nLUTHERIA_ASR_COMPUTE_TYPE=float16\nLUTHERIA_MT_DEVICE=cuda\nLUTHERIA_MT_COMPUTE_TYPE=int8\nLUTHERIA_ASR_MODEL=models/asr-mg\nLUTHERIA_MT_MODEL=models/mt-nllb\n" "$TOKEN" | sudo tee /etc/lutheria.env
   ```
6. **Nginx** : `server_name _;` OK pour IP auto-signée, mais **avant** `certbot` remplacer par domaine réel :
   ```
   sudo sed -i 's/server_name _;/server_name lutheria.duckdns.org;/' /etc/nginx/sites-available/lutheria
   # ne jamais écrire "lutheria.duckdns.org;" seul → unknown directive
   ```
7. **Outillage** : `scripts/smoke_client.py:52` omettre `ssl` quand `None` (laisser `websockets` créer le contexte par défaut pour `wss://` vérifié). `--insecure` ne crée un `ssl_context` que pour `wss://` auto-signé.

## Conséquences

- (+) Déploiement reproductible août 2026 sur AMI 26.04, coût `ADR 0006` inchangé (`g4dn.xlarge` ~0,53 $/h, 30-80 $/mois on-demand).
- (+) `certbot --nginx -d lutheria.duckdns.org` réussit (cert `fullchain.pem` `2026-11-25`, auto-renouvellement).
- (+) `smoke_client` fonctionne sans `--insecure` sur `wss://` Let’s Encrypt, avec `--insecure` sur IP auto-signée.
- (−) ADR 0006 reste historique ; le présent ADR le surcharge pour la partie AMI/Python/CUDA/réseau.
- À surveiller : `apt` `43 packages can be upgraded`, `websockets` non pinné → drift `ssl` à l’avenir.
