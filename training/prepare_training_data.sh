#!/usr/bin/env bash
set -euo pipefail

# Hugging Face dataset + file info
DATASET_ID="glab-caltech/TWIN"
BRANCH="main"
IMAGES_ZIP_NAME="images.zip"

# Local paths
DEST_DIR="training/data"
PYTHON_SCRIPT="training/prepare_training_data.py"

IMAGES_URL="https://huggingface.co/datasets/${DATASET_ID}/resolve/${BRANCH}/${IMAGES_ZIP_NAME}"
ZIP_PATH="${DEST_DIR}/${IMAGES_ZIP_NAME}"
IMAGES_DIR="${DEST_DIR}/images"

########################################
# Download images.zip
########################################

mkdir -p "${DEST_DIR}"
echo "[INFO] Downloading ${IMAGES_ZIP_NAME} from ${IMAGES_URL}"
wget -O "${ZIP_PATH}" "${IMAGES_URL}"

########################################
# Unzip into DEST_DIR/images
########################################

# Remove any existing images/ to avoid stale files
if [[ -d "${IMAGES_DIR}" ]]; then
  echo "[INFO] Removing existing images directory at ${IMAGES_DIR}"
  rm -rf "${IMAGES_DIR}"
fi

echo "[INFO] Unzipping ${ZIP_PATH} into ${DEST_DIR}"
unzip -q "${ZIP_PATH}" -d "${DEST_DIR}"

if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "[ERROR] Expected ${IMAGES_DIR} to exist after unzip, but it was not found." >&2
  exit 1
fi

echo "[INFO] Images available in ${IMAGES_DIR}"

########################################
# Run Preprocessing Python script
########################################

if [[ -f "${PYTHON_SCRIPT}" ]]; then
  echo "[INFO] Running: uv run python \"${PYTHON_SCRIPT}\""
  uv run python "${PYTHON_SCRIPT}"
else
  echo "[ERROR] Python script not found at ${PYTHON_SCRIPT}" >&2
  exit 1
fi

echo "[INFO] Done."
