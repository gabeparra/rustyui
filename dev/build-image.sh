#!/bin/sh
# Build rustyui's dev image (dev/Containerfile, context dev/), tagged by tag.sh. The image's user
# takes your user name, so its ~/.local paths are the same as yours (see the Containerfile).
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
TAG=$("$DIR/tag.sh")
podman build --build-arg DEV_USER="$(id -un)" -f "$DIR/Containerfile" -t "localhost/rustyui-dev:$TAG" "$DIR"
echo "localhost/rustyui-dev:$TAG"
