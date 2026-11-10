#!/bin/sh
# The dev image's tag: a hash of the files it is built from, so a changed pin is a new image and
# the one run is always the one these files describe
cd "$(dirname "$0")" && cat Containerfile cuda-12.9.lock | sha256sum | cut -c1-12
