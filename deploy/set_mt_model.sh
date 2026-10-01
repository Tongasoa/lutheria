#!/usr/bin/env bash
# Bascule le modèle MT de Lutheria dans /etc/lutheria.env, sans le casser.
#
# Usage :
#   sudo ./deploy/set_mt_model.sh <repertoire_mt> <tokenizer_hf>
#   sudo ./deploy/set_mt_model.sh --rollback
#
# Exemples :
#   sudo ./deploy/set_mt_model.sh models/mt-nllb facebook/nllb-200-distilled-600M
#   sudo ./deploy/set_mt_model.sh models/mt-nllb-v1 francis47/nllb_mg_v1
#   sudo ./deploy/set_mt_model.sh --rollback          # restaure la derniere sauvegarde
#
# Pourquoi un script : `EnvironmentFile` de systemd est strict. Un espace en fin
# de ligne ou un "# commentaire" inline rend la valeur littérale, et un code
# langue MT invalide ne lève rien côté tokenizer — la traduction se dégrade
# silencieusement (ADR 0008). Le script réécrit proprement les variables MT,
# refuse un modèle absent du disque, puis affiche le résultat via `cat -A`.
#
# SUDO est surchargeable (SUDO="") pour les tests ; en production c'est sudo.
set -euo pipefail

SUDO="${SUDO-sudo}"
ENV_FILE=/etc/lutheria.env
BACKUP=/etc/lutheria.env.bak
REPO=/home/ubuntu/lutheria
MT_VARS='LUTHERIA_MT_MODEL|LUTHERIA_MT_TOKENIZER_MODEL|LUTHERIA_MT_SRC_LANG|LUTHERIA_MT_TGT_LANG'

die() { echo "ERREUR: $*" >&2; exit 1; }

run() { if [ -n "$SUDO" ]; then "$SUDO" "$@"; else "$@"; fi; }

# GNU cat -A ; BSD (macOS) n'a que -et, qui montre aussi la fin de ligne.
if cat -A /dev/null >/dev/null 2>&1; then CAT_A=(-A); else CAT_A=(-et); fi

verify() {
  echo "--- $ENV_FILE (variables MT ; marqueur de fin de ligne en fin de chaque valeur) ---"
  run cat "${CAT_A[@]}" "$ENV_FILE" | grep -E "^($MT_VARS)=" || die "aucune ligne MT dans $ENV_FILE"
  echo "--- controle du piege ADR 0007 : commentaire inline ---"
  if run grep -E "^($MT_VARS)=" "$ENV_FILE" | grep -q "#"; then
    die "commentaire inline sur une variable MT : a supprimer"
  fi
  echo "aucun commentaire inline : OK"
}

case "${1:-}" in
  --rollback)
    test -f "$BACKUP" || die "aucune sauvegarde $BACKUP"
    echo "restauration de $BACKUP"
    run cp "$BACKUP" "$ENV_FILE"
    echo
    verify
    echo
    echo "appliquer :  $SUDO systemctl restart lutheria && journalctl -u lutheria -f"
    exit 0
    ;;
  "")
    die "usage : $0 <repertoire_mt> <tokenizer_hf> | --rollback"
    ;;
  *)
    MODEL="$1"; TOKENIZER="${2:-}"
    test -n "$TOKENIZER" || die "tokenizer manquant : voir l'usage"
    case "$MODEL" in /*) ;; *) MODEL="$REPO/$MODEL" ;; esac
    test -d "$MODEL" || die "modele absent du disque : $MODEL
  convertir d'abord :
  sudo -u ubuntu bash -lc 'source ~/lutheria/.venv/bin/activate && cd ~/lutheria && ./scripts/convert_ct2.sh mt <repo_hf> <sortie>'"

    run cp "$ENV_FILE" "$BACKUP"
    echo "sauvegarde -> $BACKUP"

    TMP="$(mktemp)"
    run grep -Ev "^($MT_VARS)=" "$ENV_FILE" > "$TMP" || true
    {
      cat "$TMP"
      echo "LUTHERIA_MT_MODEL=$MODEL"
      echo "LUTHERIA_MT_TOKENIZER_MODEL=$TOKENIZER"
      echo "LUTHERIA_MT_SRC_LANG=plt_Latn"
      echo "LUTHERIA_MT_TGT_LANG=fra_Latn"
    } | run tee "$ENV_FILE" >/dev/null
    rm -f "$TMP"
    echo
    echo "nouveau modele MT : $MODEL"
    echo "tokenizer        : $TOKENIZER"
    ;;
esac

echo
verify
echo
echo "appliquer       :  $SUDO systemctl restart lutheria && journalctl -u lutheria -f"
echo "retour arriere  :  $SUDO $0 --rollback && $SUDO systemctl restart lutheria"