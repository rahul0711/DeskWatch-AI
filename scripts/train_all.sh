#!/usr/bin/env bash
# Collect -> (you review) -> train phone + posture models -> switch them on.
#
#   bash scripts/train_all.sh                 # collect 8 hours, pause for review, train, enable
#   bash scripts/train_all.sh 120             # collect 120 minutes instead
#   bash scripts/train_all.sh --skip-collect  # data already collected and reviewed: train + enable
#
# The review in the middle is manual (see scripts/collect_training_data.py):
#   phones   fix the boxes in data/training/phones with Label Studio or CVAT
#   posture  sort data/training/posture/to_review/{sitting,standing}, then move
#            both folders into data/training/posture/reviewed/
# A model whose data is not ready is skipped, not trained on bad data.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
PY="$PROJECT_DIR/.venv/bin/python"
[ -x "$PY" ] || { echo ".venv missing -- run: bash deploy/setup.sh"; exit 1; }

PHONES="data/training/phones"
POSTURE="data/training/posture/reviewed"
MINUTES=480
SKIP_COLLECT=0
for arg in "$@"; do
  case "$arg" in
    --skip-collect) SKIP_COLLECT=1 ;;
    ''|*[!0-9]*) echo "Unknown argument: $arg"; exit 1 ;;
    *) MINUTES="$arg" ;;
  esac
done

# Set KEY=VALUE in .env, replacing an existing line (commented or not).
set_env() {
  local key="$1" value="$2"
  if grep -qE "^#?\s*${key}=" .env; then
    sed -i -E "s|^#?\s*${key}=.*|${key}=${value}|" .env
  else
    printf '\n%s=%s\n' "$key" "$value" >> .env
  fi
  echo "  .env: ${key}=${value}"
}

count() {
  if [ -d "$1" ]; then find "$1" -maxdepth 1 -type f -name "*.$2" | wc -l; else echo 0; fi
}

# 1. Collect --------------------------------------------------------------------
if [ "$SKIP_COLLECT" -eq 0 ]; then
  echo "== 1/3  Collecting for ${MINUTES} minutes from every camera (Ctrl+C stops early, data is kept)"
  "$PY" scripts/collect_training_data.py --minutes "$MINUTES" || true

  # 2. Review (manual) ------------------------------------------------------------
  echo
  echo "== 2/3  Review the data now (this script waits):"
  echo "   Phones : open $PHONES in Label Studio or CVAT (YOLO format), fix every box, export back"
  echo "   Posture: sort data/training/posture/to_review/{sitting,standing},"
  echo "            then move both folders into $POSTURE/"
  read -r -p "   Press Enter when the review is done (Ctrl+C to stop and run with --skip-collect later)... "
fi

# 3. Train ------------------------------------------------------------------------
echo
echo "== 3/3  Training"
trained=0

phone_imgs=$(count "$PHONES/images" jpg)
if [ "$phone_imgs" -ge 300 ]; then
  echo "-- Phone model: $phone_imgs images"
  "$PY" scripts/train_phone_model.py
  set_env ACTIVITY_PHONE_MODEL models/phone_detector.pt
  trained=1
else
  echo "-- Phone model SKIPPED: $phone_imgs reviewed images in $PHONES/images (need 300+)"
fi

sit=$(count "$POSTURE/sitting" jpg); stand=$(count "$POSTURE/standing" jpg)
if [ "$sit" -ge 300 ] && [ "$stand" -ge 300 ]; then
  echo "-- Posture model: $sit sitting, $stand standing"
  "$PY" scripts/train_posture_model.py
  set_env ACTIVITY_POSTURE_MODEL models/posture_classifier.pt
  trained=1
else
  echo "-- Posture model SKIPPED: $sit sitting / $stand standing in $POSTURE (need 300+ each)"
fi

echo
if [ "$trained" -eq 1 ]; then
  echo "Done. Restart the system to use the new model(s):  bash run_all.sh"
  echo "To switch a model off again, blank its line in .env (e.g. ACTIVITY_PHONE_MODEL=) and restart."
else
  echo "Nothing trained -- collect and review more data, then: bash scripts/train_all.sh --skip-collect"
fi
