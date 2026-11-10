#!/bin/sh
# A command in rustyui's dev container (the image tag.sh names; dev/build-image.sh makes it), as
# your user, with the GPU:
#   dev/run.sh cargo build --release
#   dev/run.sh cargo test --release -p rustyui-backend
#   dev/run.sh                      (a shell)
# The repository is mounted where it is on the host and builds go to target-dev/, so the host's own
# target/ stays as it is. A ComfyUI install (COMFYUI_DIR, default ~/ComfyUI) has its models mounted
# read-only and its input and output folders writable; rustyui's cache folder is writable.
# RUSTYUI_DEV_MOUNTS adds more read-only folders (space-separated), for example where the model
# files' symlinks point. The network is the host's: a backend started in here listens on the host's
# loopback like an installed one, so give it another --listen port while that one runs.
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DIR/.." && pwd)
IMAGE=${RUSTYUI_DEV_IMAGE:-localhost/rustyui-dev:$("$DIR/tag.sh")}
H=$HOME
C=${COMFYUI_DIR:-$H/ComfyUI}
set -- "${@}"
[ $# -eq 0 ] && set -- bash
TTY=; [ -t 0 ] && TTY=-t
mkdir -p "$H/.cache/rustyui"
MOUNTS="-v $REPO:$REPO -v $H/.cache/rustyui:$H/.cache/rustyui"
[ -d "$C/models" ] && MOUNTS="$MOUNTS -v $C/models:$C/models:ro"
for d in "$C/input" "$C/output"; do
  [ -d "$d" ] && MOUNTS="$MOUNTS -v $d:$d"
done
# RIFE's checkpoints live in their node's folder
for d in $RUSTYUI_DEV_MOUNTS "$C/custom_nodes/ComfyUI-Frame-Interpolation/ckpts"; do
  [ -d "$d" ] && MOUNTS="$MOUNTS -v $d:$d:ro"
done
exec podman run --rm -i $TTY \
  --userns=keep-id --security-opt label=disable \
  --device nvidia.com/gpu=all \
  --network host \
  -e CARGO_TARGET_DIR="$REPO/target-dev" \
  -e RUSTYUI_MODELS="$C/models" \
  -v rustyui-dev-cargo-registry:"$H/.cargo/registry" \
  -v rustyui-dev-cargo-git:"$H/.cargo/git" \
  -v rustyui-dev-cudaforge:"$H/.cudaforge" \
  $MOUNTS \
  -w "$REPO" \
  "$IMAGE" "$@"
