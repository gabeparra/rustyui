# ui: rustyui's fork of ComfyUI's editor

Copied from github.com/Comfy-Org/ComfyUI_frontend at v1.55.14 (commit 3ac9577, 2026-10-07), the
version ComfyUI's `comfyui-frontend-package==1.55.14` ships. GPL-3.0-only, like upstream.

Parts rustyui does not use were removed (the website and billing apps, browser tests, upstream's CI
and agent configs, release scripts); the list is in the root `THIRD_PARTY_NOTICES.md`.

rustyui changes go on top of this copy. To take a later upstream version: diff the new tag against
v1.55.14 and apply that diff here (git apply -3), resolving conflicts with our changes.
