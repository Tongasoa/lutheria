# Déploiement EC2 — guide pas à pas

Cible : **g4dn.xlarge**, AMI **Deep Learning Base OSS NVIDIA Driver (Ubuntu 26.04)** — fallback `24.04` si `26.04` indisponible (voir ADR 0007, amende ADR 0006),
80 Go gp3. Durée totale ~30 min la première fois.

## 1. Lancement de l'instance

1. Console AWS → EC2 → Launch instance :
   - Nom : `lutheria-prod`
   - AMI : rechercher « Deep Learning Base OSS NVIDIA Driver AMI (Ubuntu 26.04) » (fallback `24.04` si `26.04` absente)
     (publique, gratuite — seuls les coûts d'instance s'appliquent)
   - Type : `g4dn.xlarge`
   - Paire de clés SSH existante ou nouvelle
   - Security group : entrées `22` (SSH, restreint à **ton IP**), `80` (HTTP, `0.0.0.0/0` — requis pour `return 301 https` et challenge Let's Encrypt `http-01`), et `443` (HTTPS).
     Le port 8000 reste fermé : seul nginx expose (`lutheria.service:27` `127.0.0.1:8000`, ADR 0005).
   - Disque : 80 Go gp3
2. (Optionnel, tests longs) Demander une capacité Spot : Actions → Instance settings.

## 2. Premier accès + vérification GPU

```bash
ssh ubuntu@<IP_PUBLIQUE>
nvidia-smi          # doit afficher la T4
python3 --version   # 3.13 sur AMI 26.04, 3.10+ sur 22.04 (voir ADR 0007)
sudo apt update && sudo apt install -y python3.13-venv nginx libcublas12 libcublaslt12
# fallback 22.04 : sudo apt install -y python3.11-venv nginx  (pas de libcublas12 nécessaire si driver déjà complet)
ldconfig -p | grep cublas  # doit montrer libcublas.so.12 (et .13 via /opt/pytorch)
```

## 3. Récupération du code et des modèles

```bash
git clone git@github.com:Tongasoa/lutheria.git && cd lutheria
python3.13 -m venv .venv   # ou python3 -m venv .venv (3.13 sur 26.04, 3.11 sur 22.04 — code compatible 3.11–3.13)
.venv/bin/pip install -r requirements.txt
```

Modèle ASR (ton fine-tuning) :

```bash
./scripts/convert_ct2.sh asr    # -> models/asr-mg (float16 pour la T4)
```

Modèle MT :

```bash
./scripts/convert_ct2.sh mt     # -> models/mt-nllb (int8)
```

## 4. Configuration & secrets

```bash
TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
sudo install -m 600 /dev/null /etc/lutheria.env
printf "LUTHERIA_MIC_TOKEN=%s\nLUTHERIA_ASR_DEVICE=cuda\nLUTHERIA_ASR_COMPUTE_TYPE=float16\nLUTHERIA_MT_DEVICE=cuda\nLUTHERIA_MT_COMPUTE_TYPE=int8\nLUTHERIA_ASR_MODEL=models/asr-mg\nLUTHERIA_MT_MODEL=models/mt-nllb\n" "$TOKEN" | sudo tee /etc/lutheria.env >/dev/null
# ⚠️ EnvironmentFile systemd est strict : pas de "# commentaire" inline, pas de $(...), pas de guillemets.
# Mauvais : LUTHERIA_ASR_MODEL=models/asr-mg  # commentaire → valeur littérale "models/asr-mg  # commentaire" → HFValidationError (server/asr.py:39)
sudo chmod 600 /etc/lutheria.env
# noter le token affiché par : sudo grep TOKEN /etc/lutheria.env
```

## 5. Service systemd + nginx + TLS auto-signé

```bash
./scripts/gen_selfsigned_cert.sh $(curl -s ifconfig.me) deploy/certs
sudo mkdir -p /etc/nginx/certs
sudo cp deploy/certs/lutheria.crt deploy/certs/lutheria.key /etc/nginx/certs/
sudo cp deploy/nginx-lutheria.conf /etc/nginx/sites-available/lutheria
sudo ln -sf /etc/nginx/sites-available/lutheria /etc/nginx/sites-enabled/lutheria
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

sudo cp deploy/lutheria.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lutheria
journalctl -u lutheria -f        # surveiller le démarrage (chargement modèles)
```

## 6. Validation sans navigateur

Depuis ton Mac :

```bash
python scripts/smoke_client.py \
  --server wss://<IP_PUBLIQUE> --token <LE_TOKEN> --insecure \
  --file un_audio_malgache.wav
```

Attendu : lignes 🟡 (partial mg) puis 🟢 (final fr) en ~2 s après chaque phrase.

## 7. Navigateur

- Ouvrir `https://<IP>/mic.html` → accepter l'avertissement certificat (une fois) →
  saisir le token → Démarrer.
- Lecteurs : `https://<IP>/listen.html` (aucun token).

## 8. Migration Let's Encrypt (recommandé dès que des lecteurs arrivent)

1. Créer un sous-domaine gratuit [duckdns.org](https://www.duckdns.org) pointant vers l'IP (vérif `nslookup <domaine>` == `<IP>`).
2. `sudo apt install certbot python3-certbot-nginx`
3. Mettre `server_name lutheria.duckdns.org;` dans la conf nginx **avant** certbot (remplacer les deux `server_name _;` — `deploy/nginx-lutheria.conf:15,22`) :
   `sudo sed -i 's/server_name _;/server_name lutheria.duckdns.org;/' /etc/nginx/sites-available/lutheria && sudo nginx -t && sudo systemctl reload nginx`
   # ne jamais écrire "lutheria.duckdns.org;" seul → unknown directive (server_name manquant)
   `sudo certbot --nginx -d lutheria.duckdns.org`  # si "Could not find matching server block", refaire l'étape sed ci-dessus puis `certbot install --cert-name lutheria.duckdns.org`
4. Plus aucun avertissement navigateur ; renouvellement automatique.

Un vrai domaine plus tard = même procédure (ADR 0005).

## 9. Dépannage prod (ajouts 2026-08-28, voir ADR 0007)

| Symptôme | Cause | Commande |
|---|---|---|
| `journalctl -u lutheria` `HFValidationError: Repo id ... 'models/asr-mg  # ...'` | `# commentaire` inline dans `/etc/lutheria.env` (`EnvironmentFile` strict) | `sudo cat -A /etc/lutheria.env` → retirer `#` → `sudo systemctl restart lutheria` |
| `RuntimeError: Library libcublas.so.12 is not found` `server/asr.py:49` | `libcublas12` manquant sur 26.04 | `sudo apt install -y libcublas12 libcublaslt12 && ldconfig -p \| grep cublas` → `systemctl restart lutheria` |
| `ValueError: ssl=None is incompatible with a wss:// URI` `scripts/smoke_client.py:62` | `websockets 17` exige `ssl` omis sur `wss://` vérifié | Fix `scripts/smoke_client.py:52` (kwargs) ou workaround `--insecure` temporaire |
| `4401 Unauthorized` malgré bon token | `hmac.compare_digest` `server/auth.py:9` — espace/`\n` final dans `lutheria.env` | `sudo cat -A /etc/lutheria.env` → `systemctl restart lutheria` |
| `certbot Timeout Fetching http://.../.well-known` | SG `80` fermé | Ouvrir `80 0.0.0.0/0` dans le SG EC2 |
| `certbot Could not find matching server block` | `server_name _;` inchangé | `sed -i 's/server_name _;/server_name <domaine>;/' /etc/nginx/sites-available/lutheria` → `certbot install` |
| `nginx emerg unknown directive "lutheria.duckdns.org"` | `server_name` oublié | Corriger en `server_name lutheria.duckdns.org;` → `nginx -t && reload` |

## Au quotidien

| Action | Commande |
|---|---|
| Arrêter (facturation stoppée sauf disque) | `sudo shutdown now` ou console AWS |
| Redémarrer | démarrer l'instance ; IP change si pas d'Elastic IP → mettre à jour DuckDNS |
| Nouveau fine-tuning ASR | push HF → `convert_ct2.sh asr <repo>` → `systemctl restart lutheria` |
| Logs serveur | `journalctl -u lutheria -n 200` |
| Mise à jour code | `git pull && systemctl restart lutheria` |
