# dev: the pinned build container

`Containerfile` describes everything rustyui is built and tested with, each piece pinned by version
and checksum, so a build can be repeated with the same tools whatever the host has drifted to. The
pieces that decide rustyui's numbers (the CUDA toolkit, cuDNN, MKL, IPP, FFmpeg's libraries) are
the versions the exactness checks against ComfyUI were made with.

| Piece | Version | Why that one |
|---|---|---|
| base | Ubuntu 26.04, by digest | |
| apt packages | Ubuntu's archive as of the `SNAPSHOT` build argument | compilers for the C crates, cmake (sentencepiece), the `ffmpeg` program |
| CUDA toolkit | 12.9, from conda-forge, all packages listed in `cuda-12.9.lock` | what the kernels are built with, and the GCC that nvcc uses |
| cuDNN | 9.19.0.56 (the `nvidia-cudnn-cu12` wheel) | the version the reference ComfyUI runs |
| Intel MKL | 2024.2.0, static | the build torch 2.11 links (`third_party/intel-mkl`) |
| Intel IPP ICV | 2026.0.0, static | the build OpenCV 5.0.0 bundles (`third_party/intel-ippicv`) |
| FFmpeg libraries | 8.1.2, PyAV's build | what ComfyUI encodes video with (`third_party/ffmpeg-pyav`) |
| Rust | 1.99.0 | `rust-toolchain.toml` pins the same for builds outside the container |
| Node, pnpm | 22.23.1, 11.13.1 | the editors |

`cuda-12.9.lock` is conda's explicit list (the URL and md5 of each package). A changed pin is a new
image: `tag.sh` names the image by a hash of `Containerfile` and the lock, and `run.sh` runs the one
the checked-out files describe.

MKL and IPP ICV are downloaded from Intel's and OpenCV's own servers while the image builds, under
Intel's licences. The image is for building on your own machine; don't publish it, or binaries
built in it, unless Intel's terms allow it (see the root README's License section).

## Once

The scripts use rootless podman. A Docker with NVIDIA's container runtime can build and run the
same `Containerfile`.

```bash
sudo apt install podman      # or your distribution's package; the one step that needs root
dev/setup-gpu.sh             # describe the GPU to rootless podman (a CDI spec); again after a driver update
dev/build-image.sh           # about 4 GB of downloads the first time
```

`setup-gpu.sh` unpacks NVIDIA's `nvidia-ctk` into the user's home (nothing is installed system-wide),
writes a CDI spec for the GPU and driver, tells podman where to find it, and ends by running
`nvidia-smi -L` in a container to show it worked.

## Every day

```bash
dev/run.sh cargo build --release
dev/run.sh cargo test --release -p rustyui-backend
dev/run.sh target-dev/release/rustyui-backend run examples/sdxl_txt2img_api.json --out output/
dev/run.sh                   # a shell in the container
```

- The repository is mounted at the same path as on the host, and builds go to `target-dev/`, so
  the host's own `target/` is left alone.
- The container also sees the GPU, the models folder (read-only) and the output folder.
  `run.sh` lists the mounts; adjust them to where your models and outputs live.
- Cargo's registry and git checkouts are kept in named volumes, so dependencies download once.
- The network is the host's. A backend started in the container listens on the host's loopback,
  so give it another port with `--listen 127.0.0.1:PORT` if one is already running there.

A binary built in the container finds cuDNN and the CUDA libraries through rpaths at the
container's install paths. Run it with `dev/run.sh`, or on a host that has the same libraries at
the same paths.

ComfyUI itself, the reference that the captures in `backend/refs` come from, is not part of the
container. It runs in its own Python environment.
