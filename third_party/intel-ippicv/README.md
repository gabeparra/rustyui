# Intel IPP for OpenCV (ippicv 2026.0.0, static)

rustyui links Intel's Integrated Performance Primitives, in the cut-down build OpenCV bundles
(IPP ICV), for one thing: OpenCV 5.0.0 hands a uint8 bicubic resize to IPP (`ipp_resize` in
`imgproc/src/resize.cpp`), and comfyui_controlnet_aux's DepthAnythingV2Preprocessor ends with one
whenever the resolution is above the picture's shorter side (a 720-wide render at 1024). IPP's
code is closed, and a float emulation of it parted from it at about one value in a million; with
IPP itself the node matches the pack's own output exactly (`debug-depth`, and `debug-ipp-resize`
against cv2 on twelve pictures, `backend/refs/ipp_resize_ref.py`).

The version is the one in OpenCV 5.0.0's wheel (`cv2.getBuildInformation()`: "Intel IPP:
2026.0.0", the package `3rdparty/ippicv/ippicv.cmake` pins), called as OpenCV calls it:
`kernels/ipp_resize.c` makes OpenCV's Integration Wrappers call (`IwiResizeParams(0, 0, 0.75,
4)`, the border replicated), the wrappers compiled from the package's own `iw/src` as OpenCV
compiles them (`-DIW_BUILD -DICV_BASE`), and `ippInit` picks the CPU's code as OpenCV's start-up
does (`k0` on an AVX-512 CPU; `debug-ipp-resize` prints both libraries' names). Linked
statically, so the machines that run rustyui need nothing installed.

This was chosen over accepting a few depth pixels one level off. IPP ICV is not part of rustyui
and is not under rustyui's licence: you download it yourself, and building rustyui with it means
accepting Intel's terms. `EULA.txt` here is the package's licence, the Intel Simplified Software
License, and `third-party-programs.txt` its notices. A binary linked with IPP ICV should not be
redistributed under the GPL.

## Getting it (no root needed)

The package OpenCV's build downloads, from OpenCV's own third-party repository:

```bash
mkdir -p ~/.cache/rustyui/ippicv && cd ~/.cache/rustyui/ippicv
curl -fsSLO https://raw.githubusercontent.com/opencv/opencv_3rdparty/406d398c436d0465c8e53dd432d9ecd9301d5f4a/ippicv/ippicv_2026.0.0_lnx_intel64_20260327_general.tgz
echo "5198d6f76e61d5aa0fa5dde72cc22947af1a86efbf49d08b90d374699c811fd0  ippicv_2026.0.0_lnx_intel64_20260327_general.tgz" | sha256sum -c
mkdir -p ~/.local/ippicv-2026.0.0 && tar xzf ippicv_2026.0.0_lnx_intel64_20260327_general.tgz -C ~/.local/ippicv-2026.0.0
```

(OpenCV checks the same file by md5, `9a3ee0c5c3c02102faa422d60bfd1f4a`.) `build.rs` looks in
`~/.local/ippicv-2026.0.0/ippicv_lnx` unless `IPPICV_ROOT` names another copy.
