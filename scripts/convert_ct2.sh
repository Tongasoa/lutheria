#!/usr/bin/env bash
# Convertit les modèles Hugging Face au format CTranslate2 (runtime du serveur).
#
# Usage :
#   ./scripts/convert_ct2.sh asr [repo_hf] [repertoire_sortie]
#   ./scripts/convert_ct2.sh mt  [repo_hf] [repertoire_sortie]
#
# Exemples :
#   ./scripts/convert_ct2.sh asr                                          # Whisper mg par défaut
#   ./scripts/convert_ct2.sh asr Tongasoa/whisper-malagasy-medium-full-v3 # nouveau fine-tuning
#   ./scripts/convert_ct2.sh mt                                           # NLLB-600M int8
#
# Les chemins de sortie sont passés au serveur via LUTHERIA_ASR_MODEL / LUTHERIA_MT_MODEL.
set -euo pipefail

ROLE="${1:-}"
REPO=""
OUT=""
QUANT="float16"

case "$ROLE" in
  asr)
    REPO="${2:-Tongasoa/whisper-malagasy-medium-full-v2}"
    OUT="${3:-models/asr-mg}"
    ;;
  mt)
    REPO="${2:-facebook/nllb-200-distilled-600M}"
    OUT="${3:-models/mt-nllb}"
    QUANT="int8"
    ;;
  *)
    echo "Rôle requis : asr | mt" >&2
    echo "  ./scripts/convert_ct2.sh asr [repo_hf] [out]  (défaut: models/asr-mg, float16)" >&2
    echo "  ./scripts/convert_ct2.sh mt  [repo_hf] [out]  (défaut: models/mt-nllb, int8)" >&2
    exit 1
    ;;
esac

command -v ct2-transformers-converter >/dev/null 2>&1 || {
  echo "ct2-transformers-converter introuvable — installer : pip install ctranslate2 transformers" >&2
  exit 1
}

echo "[$ROLE] Conversion de $REPO -> $OUT ($QUANT)"
mkdir -p "$(dirname "$OUT")"
ct2-transformers-converter --model "$REPO" --output_dir "$OUT" --quantization "$QUANT" --force

echo "OK. Utiliser ensuite :"
if [ "$ROLE" = "asr" ]; then
  echo "  export LUTHERIA_ASR_MODEL=$OUT"
else
  echo "  export LUTHERIA_MT_MODEL=$OUT"
fi
