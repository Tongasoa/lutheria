#!/usr/bin/env bash
# Génère un certificat TLS auto-signé pour la v1 (sans nom de domaine).
#
# Usage (sur le serveur, en root ou avec sudo) :
#   sudo ./scripts/gen_selfsigned_cert.sh lutheria.local
#
# Le CN/SAN est le nom que taperont les utilisateurs ; il peut être une IP.
# Les navigateurs afficheront un avertissement unique à accepter (ADR 0005).
# Migration ultérieure vers Let's Encrypt : voir deploy/DEPLOYMENT.md §6.
set -euo pipefail

NAME="${1:-lutheria.local}"
OUT_DIR="${2:-deploy/certs}"

mkdir -p "$OUT_DIR"

openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
  -keyout "$OUT_DIR/lutheria.key" \
  -out "$OUT_DIR/lutheria.crt" \
  -subj "/CN=$NAME" \
  -addext "subjectAltName=DNS:$NAME,IP:$(dig +short "$NAME" 2>/dev/null || echo 127.0.0.1)"

chmod 600 "$OUT_DIR/lutheria.key"
chmod 644 "$OUT_DIR/lutheria.crt"

echo "Certificat généré :"
echo "  $OUT_DIR/lutheria.crt"
echo "  $OUT_DIR/lutheria.key"
echo "N'oubliez pas : ce répertoire est ignoré par git (.gitignore)."
