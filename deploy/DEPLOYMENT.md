# Déploiement EC2 — guide pas à pas

Cible : **g4dn.xlarge**, AMI **Deep Learning Base OSS NVIDIA Driver (Ubuntu 22.04)**,
80 Go gp3 (voir ADR 0006). Durée totale ~30 min la première fois.

## 1. Lancement de l'instance

1. Console AWS → EC2 → Launch instance :
   - Nom : `lutheria-prod`
   - AMI : rechercher « Deep Learning Base OSS NVIDIA Driver AMI (Ubuntu 22.04) »
     (publique, gratuite — seuls les coûts d'instance s'appliquent)
   - Type : `g4dn.xlarge`
   - Paire de clés SSH existante ou nouvelle
   - Security group : entrées `22` (SSH, restreint à **ton IP**) et `443` (HTTPS).
     Le port 8000 reste fermé : seul nginx expose.
   - Disque : 80 Go gp3
2. (Optionnel, tests longs) Demander une capacité Spot : Actions → Instance settings.

## 2. Premier accès + vérification GPU

```bash
ssh ubuntu@<IP_PUBLIQUE>
nvidia-smi          # doit afficher la T4
python3 --version   # 3.10+ sur l'AMI DL (le venv du projet utilise >=3.11 sinon apt install python3.11-venv)
sudo apt update && sudo apt install -y python3.11-venv nginx
```

## 3. Récupération du code et des modèles

```bash
git clone git@github.com:Tongasoa/lutheria.git && cd lutheria
python3.11 -m venv .venv
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
sudo install -m 600 /dev/null /etc/lutheria.env
cat <<EOF | sudo tee /etc/lutheria.env >/dev/null
LUTHERIA_MIC_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
LUTHERIA_ASR_DEVICE=cuda
LUTHERIA_ASR_COMPUTE_TYPE=float16
LUTHERIA_MT_DEVICE=cuda
LUTHERIA_MT_COMPUTE_TYPE=int8
LUTHERIA_ASR_MODEL=models/asr-mg
LUTHERIA_MT_MODEL=models/mt-nllb
EOF
# noter le token affiqué par : sudo grep TOKEN /etc/lutheria.env
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

1. Créer un sous-domaine gratuit [duckdns.org](https://www.duckdns.org) pointant vers l'IP.
2. `sudo apt install certbot python3-certbot-nginx`
3. Mettre `server_name lutheria.duckdns.org;` dans la conf nginx puis :
   `sudo certbot --nginx -d lutheria.duckdns.org`
4. Plus aucun avertissement navigateur ; renouvellement automatique.

Un vrai domaine plus tard = même procédure (ADR 0005).

## Au quotidien

| Action | Commande |
|---|---|
| Arrêter (facturation stoppée sauf disque) | `sudo shutdown now` ou console AWS |
| Redémarrer | démarrer l'instance ; IP change si pas d'Elastic IP → mettre à jour DuckDNS |
| Nouveau fine-tuning ASR | push HF → `convert_ct2.sh asr <repo>` → `systemctl restart lutheria` |
| Logs serveur | `journalctl -u lutheria -n 200` |
| Mise à jour code | `git pull && systemctl restart lutheria` |
