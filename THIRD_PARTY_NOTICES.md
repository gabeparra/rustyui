# Third-party notices

rustyui is licensed under the GNU General Public License v3.0 (see `LICENSE`). It includes, ports or
builds on the following works, whose licenses are reproduced or referenced here.

## ComfyUI (GPL-3.0)

Most of the backend follows ComfyUI's code closely, among others: the prompt language
(`backend/src/prompt.rs`, from `comfy/sd1_clip.py`), the samplers and schedules
(`backend/src/sampling.rs`, `backend/src/brownian.rs`), tiled upscaling (`backend/src/upscale.rs`),
the LTX-2.3 model and nodes (`backend/src/ltx*.rs`, `graph_ltx.rs`, from `comfy/ldm/lightricks` and
`comfy_extras/nodes_lt.py`), Anima (`backend/src/anima*.rs`, `qwen3.rs`) and the Wan 2.1 VAE.

## ComfyUI frontend (GPL-3.0)

`ui/` is a copy of https://github.com/Comfy-Org/ComfyUI_frontend at v1.55.14, licensed under the
GNU General Public License v3.0 (`ui/LICENSE`), with its own third-party notices in
`ui/THIRD_PARTY_NOTICES.md`. `ui/UPSTREAM.md` records the upstream version.

The copy is v1.55.14 with the parts rustyui does not build or serve removed: `apps/` (the
marketing website and the billing app), `browser_tests/`, upstream's `.github/`, `.claude/`,
`.agents/`, `.vscode/`, `.storybook/`, `.husky/` and other repository tooling configs (CODEOWNERS,
CodeRabbit, MCP, Playwright configs), `scripts/cicd`, `scripts/registry-census`,
`scripts/release-sheriff` and `scripts/skills`, `tools/devtools`, `tools/test-recorder`, a 12 MB
demo gif of a third-party project, two unit tests that depended on the removed browser tests, and
the Algolia key in `.env_example`. The workspace entries, root scripts and `pnpm-lock.yaml` were
updated to match. `ui/LICENSE` and `ui/THIRD_PARTY_NOTICES.md` are kept as upstream has them.

ComfyUI is developed by comfyanonymous and
contributors and licensed under the GNU General Public License v3.0:
https://github.com/comfyanonymous/ComfyUI. Its samplers in turn come from k-diffusion by Katherine
Crowson (MIT): https://github.com/crowsonkb/k-diffusion

## candle (MIT OR Apache-2.0)

`third_party/candle-core` and `third_party/candle-flash-attn` are modified copies of crates from
https://github.com/huggingface/candle, available under either the MIT license or the Apache License
2.0. Both texts are in each crate's folder (`LICENSE-MIT`, `LICENSE`). The modifications are marked
with `rustyui:` comments. `backend/src/qgemma3.rs` is a modified copy of candle-transformers'
`quantized_gemma3.rs`, under the same terms.

## FlashAttention (BSD-3-Clause)

`third_party/candle-flash-attn/kernels/` holds FlashAttention-2 kernels from
https://github.com/Dao-AILab/flash-attention (Copyright (c) Tri Dao), trimmed and extended with
split-KV instantiations. CUTLASS (BSD-3-Clause, NVIDIA) is downloaded at build time.

```
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## CUTLASS, cuDNN frontend (BSD-3-Clause, MIT)

`third_party/cutlass` holds the CUTLASS headers at the commit PyTorch 2.11 pins (BSD-3-Clause,
NVIDIA, `third_party/cutlass/LICENSE.txt`). `third_party/cudnn-frontend` holds NVIDIA's cuDNN
frontend headers (MIT, `third_party/cudnn-frontend/LICENSE.txt`).

## comfy_kitchen (Apache-2.0)

`backend/kernels/ck.cu` carries CUDA kernels from comfy_kitchen 0.2.37, Copyright (c) 2025 Comfy Org
and NVIDIA CORPORATION & AFFILIATES, under the Apache License 2.0 (`backend/kernels/LICENSE-comfy-kitchen`).
The file says what was changed.

## PyTorch (BSD-3-Clause)

`backend/src/rng.rs` and `backend/src/torchcpu.rs` reproduce PyTorch's random number generators and
CPU math, `backend/kernels/ops.cu` and `backend/src/kernels.rs` follow the arithmetic of PyTorch's
CUDA kernels, and `third_party/torch-mem-eff` holds PyTorch's memory-efficient attention headers,
unchanged apart from two shims (`third_party/torch-mem-eff/README.md`).

```
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## avx_mathfun (zlib)

The CPU `randn` in `backend/src/rng.rs` reproduces, operation for operation, the AVX `log` and
`sincos` of PyTorch's `avx_mathfun.h`. This is an altered version of that work.

```
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
```

## Pillow (MIT-CMU)

`backend/src/resample.rs` is a port of Pillow's `Resample.c` (lanczos resampling).

```
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

## Custom node packs and models

rustyui reimplements nodes from these ComfyUI node packs and models, following their code. The
licences below are the ones their repositories state; check them before redistributing.

| Project | Used for | Licence |
|---|---|---|
| ComfyUI-Impact-Pack | detailer and SEGS nodes (`impact.rs`, `graph_impact.rs`) | GPL-3.0 |
| ComfyUI-KJNodes | ImageResizeKJv2 and other utility nodes | GPL-3.0 |
| comfyui_controlnet_aux | DepthAnythingV2Preprocessor, Canny | Apache-2.0 |
| ComfyUI-RMBG | SAM3Segment | GPL-3.0 |
| ComfyUI-Frame-Interpolation, RIFE | RIFE VFI (`rife.rs`) | MIT |
| ComfyUI-TiledAttentionCouple | the attention couple nodes (`graph_tac.rs`) | see its repository |
| Depth Anything V2 | the depth model (`depth_anything.rs`) | code Apache-2.0; most checkpoints CC BY-NC 4.0 |
| SAM 3 | the segmentation model (`sam3.rs`) | Meta's SAM License |

No model weights are included. `backend/assets/anima/` holds tokenizer files built from Qwen's
vocabulary and merges (Apache-2.0) and Google's T5 vocabulary (Apache-2.0); `backend/assets/clip/`
holds CLIP's BPE vocabulary (MIT, OpenAI).

## Intel MKL and IPP ICV (not included)

The build links Intel oneAPI MKL 2024.2.0 and Intel IPP ICV 2026.0.0, which you download yourself
(`third_party/intel-mkl/README.md`, `third_party/intel-ippicv/README.md`). They are under Intel's
own licences (the Intel Simplified Software License and the Intel End User License Agreement for
Developer Tools), not the GPL. Building rustyui with them means accepting those terms, and a binary
linked with them should not be redistributed under the GPL.

## FFmpeg (not included)

Videos are encoded by running FFmpeg (GPL-2.0-or-later with x264) as a separate program, optionally
on PyAV's FFmpeg libraries (`third_party/ffmpeg-pyav/README.md`).

## Other dependencies

The Rust crates rustyui depends on are listed in `Cargo.lock`; all are under permissive licenses
(MIT, Apache-2.0, BSD, ISC, Zlib, Unicode-3.0 or the Unlicense). The simple editor bundles
Vue (MIT) and @comfyorg/litegraph (MIT); `ui/` lists its own.
