#!/usr/bin/env bash
# Convertit un modèle Whisper Hugging Face au format CTranslate2 (faster-whisper).
#
# Usage :
#   ./scripts/convert_ct2.sh [repo_hf_ou_chemin] [repertoire_sortie]
#
# Exemples :
#   ./scripts/convert_ct2.sh                                          # modèle par défaut
#   ./scripts/convert_ct2.sh Tongasoa/whisper-malagasy-medium-full-v3 # nouveau fine-tuning
#
# Le chemin de sortie est ensuite passé à LUTHERIA_ASR_MODEL côté serveur.
set -euo pipefail

REPO="${1:-Tongasoa/whisper-malagasy-medium-full-v2}"
OUT="${2:-models/asr-mg}"

command -v ct2-transformers-converter >/dev/null 2>&1 || {
  echo "ct2-transformers-converter introuvable — installer ctranslate2 : pip install ctranslate2 transformers" >&2
  exit 1
}

echo "Conversion de $REPO -> $OUT (float16)"
mkdir -p "$(dirname "$OUT")"
ct2-transformers-converter --model "$REPO" --output_dir "$OUT" --quantization float16 --force

echo "OK. Utiliser ensuite :"
echo "  export LUTHERIA_ASR_MODEL=$OUT"
