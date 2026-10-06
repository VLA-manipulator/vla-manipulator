#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${1:?usage: archive_checkpoint.sh CHECKPOINT_DIR HF_MODEL_REPO}"
HF_MODEL_REPO="${2:?usage: archive_checkpoint.sh CHECKPOINT_DIR HF_MODEL_REPO}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/root/autodl-tmp/checkpoint-archives}"

checkpoint_abs="$(readlink -f "${CHECKPOINT_DIR}")"
case "${checkpoint_abs}" in
  /root/autodl-tmp/checkpoints/*) ;;
  *) echo "Refusing to archive unexpected checkpoint path: ${checkpoint_abs}" >&2; exit 1 ;;
esac

mkdir -p "${ARCHIVE_ROOT}"
archive="${ARCHIVE_ROOT}/$(basename "$(dirname "${checkpoint_abs}")")-$(basename "${checkpoint_abs}").tar.zst"
tar --zstd -cf "${archive}" -C "$(dirname "${checkpoint_abs}")" "$(basename "${checkpoint_abs}")"
hf upload "${HF_MODEL_REPO}" "${archive}" "checkpoints/$(basename "${archive}")" --repo-type model
echo "Uploaded ${archive}. Verify it on Hugging Face before deleting any local checkpoint."
