// Compile rustyui's CUDA and C++ kernels, link the native libraries whose arithmetic ComfyUI's
// results depend on (cuDNN, Intel MKL's VML, Intel IPP ICV), let the binary find a CUDA toolkit
// and cuDNN outside the system library path without LD_LIBRARY_PATH, and compile the editors
// into the binary: ComfyUI's, forked (../ui/dist), and the simple editor (../frontend/dist).
//
// Where things are looked for (each can be set in the environment):
//   CUDA_ROOT        the CUDA 12.9 toolkit: else CUDA_PATH, the nvcc on PATH, ~/.local/cuda-12.9
//                    (the dev container's layout) or /usr/local/cuda
//   CUDNN_ROOT       cuDNN 9.19: else ~/.local/cudnn-9.19.0-cu12, or the CUDA toolkit's folder
//   RUSTYUI_HOST_CXX the host C++ compiler for nvcc and the C++ kernels: else the conda g++ inside
//                    CUDA_ROOT when there is one, or g++
//   MKLROOT          Intel MKL 2024.2.0 (third_party/intel-mkl/README.md)
//   IPPICV_ROOT      Intel IPP ICV 2026.0.0 (third_party/intel-ippicv/README.md)
use std::path::{Path, PathBuf};

/// `~/.local/<rel>`, the dev container's install prefix
fn home_local(rel: &str) -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local").join(rel)
}

fn env_dir(var: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={var}");
    std::env::var(var).ok().filter(|v| !v.is_empty())
}

fn cuda_root() -> String {
    if let Some(d) = env_dir("CUDA_ROOT").or_else(|| env_dir("CUDA_PATH")) {
        return d;
    }
    let on_path = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
        .find(|dir| dir.join("nvcc").is_file());
    if let Some(root) = on_path.as_deref().and_then(Path::parent) {
        return root.to_string_lossy().into_owned();
    }
    let local = home_local("cuda-12.9");
    if local.join("bin/nvcc").is_file() {
        return local.to_string_lossy().into_owned();
    }
    "/usr/local/cuda".into()
}

/// The toolkit's library folder: lib (a conda prefix) or lib64 (NVIDIA's installers)
fn lib_dir(root: &str) -> String {
    if Path::new(root).join("lib64").is_dir() && !Path::new(root).join("lib").is_dir() {
        format!("{root}/lib64")
    } else {
        format!("{root}/lib")
    }
}

fn main() {
    let cuda = cuda_root();
    let cuda_lib = lib_dir(&cuda);
    let cxx = env_dir("RUSTYUI_HOST_CXX").unwrap_or_else(|| {
        let conda = format!("{cuda}/bin/x86_64-conda-linux-gnu-g++");
        if Path::new(&conda).is_file() {
            conda
        } else {
            "g++".into()
        }
    });
    // cuDNN 9.19, the version the reference ComfyUI runs (torch 2.11 cu130): its runtime-compiled
    // attention kernels are what a masked attention in ComfyUI runs, and its convolutions are
    // ComfyUI's too. Its rpath comes first, so it is the libcudnn.so.9 candle loads as well.
    let cudnn = env_dir("CUDNN_ROOT").unwrap_or_else(|| {
        let local = home_local("cudnn-9.19.0-cu12");
        if local.is_dir() {
            local.to_string_lossy().into_owned()
        } else {
            cuda.clone()
        }
    });
    let cudnn_lib = lib_dir(&cudnn);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cudnn_lib}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda_lib}");
    // where kernels.rs loads NVRTC from (cuDNN dlopens it by name, and nothing else puts it on the
    // loader's path)
    println!("cargo:rustc-env=RUSTYUI_CUDA_LIB={cuda_lib}");
    println!("cargo:rerun-if-changed=kernels/ops.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".into());
    let out = std::env::var("OUT_DIR").unwrap();
    let status = std::process::Command::new(format!("{cuda}/bin/nvcc"))
        .args(["--ptx", "-O3", "-std=c++17", &format!("-arch=sm_{cap}"), "-o", &format!("{out}/ops.ptx"), "kernels/ops.cu"])
        .status()
        .expect("running nvcc");
    assert!(status.success(), "nvcc failed on kernels/ops.cu");
    // comfy_kitchen's kernels, with the flags comfy_kitchen builds them with (fast math: its
    // rsqrtf and divisions round as ComfyUI's do only then)
    println!("cargo:rerun-if-changed=kernels/ck.cu");
    let status = std::process::Command::new(format!("{cuda}/bin/nvcc"))
        .args([
            "--ptx",
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            &format!("-arch=sm_{cap}"),
            "-o",
            &format!("{out}/ck.ptx"),
            "kernels/ck.cu",
        ])
        .status()
        .expect("running nvcc");
    assert!(status.success(), "nvcc failed on kernels/ck.cu");
    // torch's memory-efficient attention forward (kernels/mem_eff.cu on third_party/torch-mem-eff and
    // third_party/cutlass at torch's CUTLASS commit), built as torch builds it: no fast math, its
    // CUTLASS defines, conda's g++ for the host side
    println!("cargo:rerun-if-changed=kernels/mem_eff.cu");
    println!("cargo:rerun-if-changed=../third_party/torch-mem-eff");
    println!("cargo:rerun-if-changed=../third_party/cutlass");
    let obj = format!("{out}/mem_eff.o");
    let status = std::process::Command::new(format!("{cuda}/bin/nvcc"))
        .args([
            "-c",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES=1",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            &format!("-arch=sm_{cap}"),
            "-Xcompiler",
            "-fPIC",
            "-ccbin",
            &cxx,
            "-I../third_party/torch-mem-eff/include",
            "-I../third_party/cutlass/include",
            "-o",
            &obj,
            "kernels/mem_eff.cu",
        ])
        .status()
        .expect("running nvcc");
    assert!(status.success(), "nvcc failed on kernels/mem_eff.cu");
    let lib = format!("{out}/librustyui_mem_eff.a");
    let _ = std::fs::remove_file(&lib);
    let status = std::process::Command::new("ar").args(["crs", &lib, &obj]).status().expect("running ar");
    assert!(status.success(), "ar failed on mem_eff.o");
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=rustyui_mem_eff");
    // cuDNN's attention as torch builds it (kernels/sdpa.cpp on third_party/cudnn-frontend), with
    // conda's g++ like nvcc's host code: the binary runs on conda's libstdc++
    println!("cargo:rerun-if-changed=kernels/sdpa.cpp");
    println!("cargo:rerun-if-changed=kernels/conv.cpp");
    println!("cargo:rerun-if-changed=../third_party/cudnn-frontend/include");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .compiler(&cxx)
        .opt_level(2)
        .warnings(false)
        .include("../third_party/cudnn-frontend/include")
        .include(format!("{cudnn}/include"))
        .include(format!("{cuda}/targets/x86_64-linux/include"))
        .file("kernels/sdpa.cpp")
        .file("kernels/conv.cpp")
        .compile("rustyui_cudnn");
    println!("cargo:rustc-link-search=native={cudnn_lib}");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // Intel's MKL 2024.2.0, the build torch 2.11 links (torch.__config__: "Version 2024.2-Product
    // Build 20240605"), linked statically as torch links it, so the binary carries it and no box
    // needs it installed: torch's CPU float exp and log are MKL's VML (aten/src/ATen/cpu/vml.h), and
    // ComfyUI's normal and sgm_uniform schedules go through them (torchcpu.rs). LP64 interface,
    // sequential layer (VML on one value or a thousand runs no threads). Unpacked from Intel's
    // oneAPI APT packages (third_party/intel-mkl/README.md); MKLROOT names another copy.
    let mkl =
        env_dir("MKLROOT").unwrap_or_else(|| home_local("intel-mkl-2024.2.0/opt/intel/oneapi/mkl/2024.2").to_string_lossy().into_owned());
    let lib = format!("{mkl}/lib");
    for a in ["libmkl_intel_lp64.a", "libmkl_sequential.a", "libmkl_core.a"] {
        assert!(
            Path::new(&lib).join(a).exists(),
            "{lib}/{a} is missing: unpack MKL 2024.2.0 as third_party/intel-mkl/README.md says, or set MKLROOT"
        );
    }
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    for a in ["libmkl_intel_lp64.a", "libmkl_sequential.a", "libmkl_core.a"] {
        println!("cargo:rustc-link-arg={lib}/{a}");
    }
    println!("cargo:rustc-link-arg=-Wl,--end-group");
    // Intel IPP's ICV build, the one OpenCV 5.0.0 bundles (ippicv 2026.0.0), linked statically, and
    // its Integration Wrappers compiled as OpenCV compiles them (-DIW_BUILD -DICV_BASE): OpenCV hands
    // a uint8 bicubic resize to it, and Depth Anything's preprocessor ends with one
    // (kernels/ipp_resize.c, third_party/intel-ippicv/README.md); IPPICV_ROOT names another copy.
    let ipp = env_dir("IPPICV_ROOT").unwrap_or_else(|| home_local("ippicv-2026.0.0/ippicv_lnx").to_string_lossy().into_owned());
    println!("cargo:rerun-if-changed=kernels/ipp_resize.c");
    let (icv, iw) = (format!("{ipp}/icv"), format!("{ipp}/iw"));
    assert!(
        Path::new(&format!("{icv}/lib/intel64/libippicv.a")).exists(),
        "{icv}/lib/intel64/libippicv.a is missing: unpack ippicv 2026.0.0 as third_party/intel-ippicv/README.md says, or set IPPICV_ROOT"
    );
    let mut iw_c = cc::Build::new();
    iw_c.define("IW_BUILD", None)
        .define("ICV_BASE", None)
        .include(format!("{icv}/include"))
        .include(format!("{iw}/include"))
        .opt_level(2)
        .warnings(false);
    let mut srcs: Vec<_> = std::fs::read_dir(format!("{iw}/src"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "c"))
        .collect();
    srcs.sort();
    iw_c.files(srcs).file("kernels/ipp_resize.c").compile("rustyui_ippiw");
    cc::Build::new()
        .cpp(true)
        .compiler(&cxx)
        .define("IW_BUILD", None)
        .define("ICV_BASE", None)
        .include(format!("{icv}/include"))
        .include(format!("{iw}/include"))
        .opt_level(2)
        .warnings(false)
        .file(format!("{iw}/src/iw_own.cpp"))
        .compile("rustyui_ippiw_cpp");
    println!("cargo:rustc-link-search=native={icv}/lib/intel64");
    println!("cargo:rustc-link-lib=static=ippicv");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=dl");
    embed(
        &out,
        "editor.rs",
        "../ui/dist",
        "EDITOR",
        "cd ui && pnpm install --frozen-lockfile && pnpm exec vite build --config vite.config.mts (ui/UPSTREAM.md)",
    );
    embed(&out, "simple.rs", "../frontend/dist", "SIMPLE", "cd frontend && bun install --frozen-lockfile && bun run build");
}

/// `$OUT_DIR/<file>`: every file of `dir` as (path, bytes) in a static named `name`, the
/// sourcemaps left out. Without a built editor the backend still builds (the API is what the
/// clients need), and the editor's page says how to build it.
fn embed(out: &str, file: &str, dir: &str, name: &str, how: &str) {
    let dist = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(dir);
    println!("cargo:rerun-if-changed={dir}");
    let mut files = vec![];
    walk(&dist, &dist, &mut files);
    files.retain(|(rel, _)| !rel.ends_with(".map"));
    files.sort();
    if !files.iter().any(|(rel, _)| rel == "index.html") {
        println!("cargo:warning={dir} has no index.html, so this backend lacks that editor: {how}, then build again");
    }
    let mut src = format!("/// {dir}, by path\npub static {name}: &[(&str, &[u8])] = &[\n");
    for (rel, abs) in &files {
        src += &format!("    ({rel:?}, include_bytes!({abs:?})),\n");
    }
    src += "];\n";
    std::fs::write(format!("{out}/{file}"), src).unwrap();
}

fn walk(root: &Path, dir: &Path, files: &mut Vec<(String, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(root, &p, files);
        } else if p.is_file() {
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            files.push((rel, p.canonicalize().unwrap().to_string_lossy().to_string()));
        }
    }
}
