#!/bin/sh
# Once, after podman is installed (sudo apt install podman): NVIDIA's nvidia-ctk, unpacked from its
# package (no root), describes this box's card and driver as a CDI spec in ~/.config/cdi, and
# podman is told to read specs there, so `--device nvidia.com/gpu=all` gives a container the card.
# Run it again after a driver update.
set -e
V=1.20.1-1
D=$HOME/.local/nvidia-ctk-$V
if [ ! -x "$D/usr/bin/nvidia-ctk" ]; then
  T=$(mktemp -d)
  curl -fsSLo "$T/base.deb" "https://nvidia.github.io/libnvidia-container/stable/deb/amd64/nvidia-container-toolkit-base_${V}_amd64.deb"
  echo "e01964a064fbdb62425ab22ec5832349f92c5b0b50c8f23fdf14dad004db4fb2  $T/base.deb" | sha256sum -c
  dpkg-deb -x "$T/base.deb" "$D"
  rm -rf "$T"
fi
mkdir -p "$HOME/.config/cdi" "$HOME/.config/containers"
# the spec's hooks run the hook binary unpacked beside nvidia-ctk (nothing is installed in /usr/bin)
"$D/usr/bin/nvidia-ctk" cdi generate --nvidia-cdi-hook-path="$D/usr/bin/nvidia-cdi-hook" --output="$HOME/.config/cdi/nvidia.yaml"
CONF=$HOME/.config/containers/containers.conf
if grep -q cdi_spec_dirs "$CONF" 2>/dev/null; then
  :
elif grep -q '^\[engine\]' "$CONF" 2>/dev/null; then
  echo "$CONF has an [engine] section: add cdi_spec_dirs = [\"$HOME/.config/cdi\", \"/etc/cdi\", \"/var/run/cdi\"] to it" >&2
  exit 1
else
  printf '[engine]\ncdi_spec_dirs = ["%s/.config/cdi", "/etc/cdi", "/var/run/cdi"]\n' "$HOME" >> "$CONF"
fi
podman run --rm --device nvidia.com/gpu=all docker.io/library/ubuntu:26.04@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7 nvidia-smi -L
