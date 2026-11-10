# FFmpeg as PyAV builds it (pyav-ffmpeg 8.1.3-1)

**Version note.** ComfyUI's PyAV 18.1.0 carries FFmpeg 8.1.2 (pyav-ffmpeg `8.1.2-1`), and everything
below was measured with that build. The PyAV project withdrew the `8.1.2-1` release in November
2026, so rustyui now fetches `8.1.3-1`, the next patch release. The packet-for-packet comparison
with ComfyUI has not been repeated with 8.1.3 yet.

ComfyUI writes its videos through PyAV, and PyAV 18.1.0 carries its own FFmpeg: 8.1.2 with
x264 (core 165), built by the PyAV project (github.com/PyAV-Org/pyav-ffmpeg, release `8.1.2-1`).
rustyui runs the system's `ffmpeg` program on those same libraries (`ffmpeg()` in
`backend/src/ltx_video.rs` puts them first on `LD_LIBRARY_PATH`), which FFmpeg's ABI allows
within a major version (the program is 8.0.1, the libraries 8.1.2).

Why: the system's FFmpeg 8.0.1 AAC encoder writes other packets than 8.1.2 for some sound. For
`ltx_av_i2v.py`'s default graph the two parted at packet 199 of 221, with the same waveform in;
with these libraries rustyui's ffmpeg command writes ComfyUI's packets exactly. The H.264 side
was already the same with either.

The libraries are the ones inside ComfyUI's PyAV wheel: the `.text` of `libavcodec.so.62.28.102`
and `libx264.so.165` here hashes the same as `av.libs/libavcodec-5e6b81c2.so.62.28.102` and
`av.libs/libx264-d6533a8d.so.165` there (auditwheel only renames them). Nothing of Python is used.

## Getting it

```bash
curl -sLO https://github.com/PyAV-Org/pyav-ffmpeg/releases/download/8.1.3-1/ffmpeg-manylinux-x86_64.tar.gz
echo "4d59bd414a731e2876c0b4cc88f5435385904727d47b0b7da5c5530048423099  ffmpeg-manylinux-x86_64.tar.gz" | sha256sum -c
mkdir -p ~/.local/ffmpeg-8.1.3-pyav && tar xzf ffmpeg-manylinux-x86_64.tar.gz -C ~/.local/ffmpeg-8.1.3-pyav lib
```

The checksum is the one GitHub publishes for the release asset. `RUSTYUI_FFMPEG_LIB` names
another folder of them. Without them rustyui uses the system's FFmpeg and says so once.

Licence: FFmpeg built with x264 is GPL-2.0-or-later (FFmpeg's `LICENSE.md`, x264's `COPYING`);
the build scripts are in the pyav-ffmpeg repository. The tarball itself carries no licence file.
