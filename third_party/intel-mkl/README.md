# Intel oneAPI MKL 2024.2.0 (static, VML only)

rustyui links Intel's Math Kernel Library statically for one thing: torch's CPU float `exp` and
`log`, which torch hands to MKL's vector math (VML: `vmsExp`, `vmsLn`, `vmdLn` with
`VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE`, `aten/src/ATen/cpu/vml.h`). ComfyUI's sigma table
and its normal and sgm_uniform schedules go through them, and MKL's results are not a correctly
rounded exp (about 1% of floats come out an ulp away), so nothing else gives the same sigmas.
With it, all four schedulers match ComfyUI for every step count from 1 to 300
(`backend/refs/cmp_schedule.py`).

The version is the one torch 2.11 is built with (`torch.__config__.show()`: "Intel(R) oneAPI
Math Kernel Library Version 2024.2-Product Build 20240605"), linked the way torch links it:
statically, LP64 interface. rustyui uses the sequential layer; VML on a single value or the
1000-value sigma table runs no threads either way. The binary carries what it uses (about
0.7 MB), so the machines that run it need nothing installed.

This was chosen over shipping a table of torch's values. MKL is not part of rustyui and is not
under rustyui's licence: you download it yourself, and building rustyui with it means accepting
Intel's terms. `license.htm` here is the licence file of the oneAPI 2024.2 packages, the Intel End
User License Agreement for Developer Tools (April 2023) and the Intel Simplified Software License
(October 2022). A binary linked with MKL should not be redistributed under the GPL.

## Getting it (no root needed)

From Intel's oneAPI APT repository, unpacked into a user prefix; `build.rs` looks in
`~/.local/intel-mkl-2024.2.0/opt/intel/oneapi/mkl/2024.2` unless `MKLROOT` names another copy.

```bash
mkdir -p ~/.cache/rustyui/intel-mkl-debs && cd ~/.cache/rustyui/intel-mkl-debs
B=https://apt.repos.intel.com/oneapi/pool/main
for f in intel-oneapi-mkl-core-2024.2-2024.2.0-663_amd64.deb \
         intel-oneapi-mkl-core-devel-2024.2-2024.2.0-663_amd64.deb \
         intel-oneapi-mkl-classic-include-2024.2-2024.2.0-663_amd64.deb \
         intel-oneapi-mkl-classic-include-common-2024.2-2024.2.0-663_all.deb \
         intel-oneapi-common-licensing-2024.2-2024.2.0-589_all.deb; do curl -sO "$B/$f"; done
sha256sum -c <<'EOF'
970f620c81941e738000bb1d247138512001951cf70fd3f7989c29bc6f267d16  intel-oneapi-mkl-core-2024.2-2024.2.0-663_amd64.deb
f2ab7f78792b078896c8f46e26eaafddd040c5a952dd5d7785c766c9f790ac94  intel-oneapi-mkl-core-devel-2024.2-2024.2.0-663_amd64.deb
c94cdf606963b0d2329ce9eebabf248cc1e4aaea55849a5477efd9eda0a68947  intel-oneapi-mkl-classic-include-2024.2-2024.2.0-663_amd64.deb
7091ad52eb772c3e1e85c10f00d25d6cd05206dc4b2f21e0560a881cf7f318cf  intel-oneapi-mkl-classic-include-common-2024.2-2024.2.0-663_all.deb
f4e94cbbacd38d7e75067419d53a71cbb9de8fdac6fdd438977d2f6fc0e87e4b  intel-oneapi-common-licensing-2024.2-2024.2.0-589_all.deb
EOF
for f in *.deb; do dpkg-deb -x "$f" ~/.local/intel-mkl-2024.2.0; done
```

The checksums are the repository index's (`dists/all/main/binary-amd64/Packages.gz` and
`binary-all/Packages.gz`). Only `lib/libmkl_intel_lp64.a`, `libmkl_sequential.a` and
`libmkl_core.a` are linked; the headers were read for the function signatures and mode bits,
which `backend/src/torchcpu.rs` declares itself.
