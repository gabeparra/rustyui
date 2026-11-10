//! Names that arrive in a request or a graph (a model, a filename_prefix, a /view filename),
//! kept inside the folder they are joined to.

use std::path::{Component, Path, PathBuf};

/// The models folder, laid out as ComfyUI's (`checkpoints/`, `loras/`, `embeddings/`, ...):
/// `RUSTYUI_MODELS` (set by `--models`), else `models` in the working directory.
pub fn models_dir() -> PathBuf {
    std::env::var_os("RUSTYUI_MODELS").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("models"))
}

/// The output folder: `RUSTYUI_OUTPUT` (set by `--out`), else `output` in the working directory.
pub fn output_dir() -> PathBuf {
    std::env::var_os("RUSTYUI_OUTPUT").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("output"))
}

/// `name` as a path that stays below whatever folder it is joined to: relative, no `..`, no
/// root. `Path::join` with an absolute name replaces the folder outright, so a check for
/// ".." alone is not enough. ComfyUI's folder_paths and SaveImage hold names to the same rule.
pub fn contained(name: &str) -> Option<PathBuf> {
    let p = Path::new(name);
    let plain = p.components().all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    (plain && p.components().any(|c| matches!(c, Component::Normal(_)))).then(|| p.to_path_buf())
}

/// Whether `path` (which need not exist yet) is below `base` once symlinks are followed:
/// the deepest part of it that exists is resolved and compared, so nothing is created first.
pub fn below(base: &Path, path: &Path) -> bool {
    let Ok(base) = base.canonicalize() else { return false };
    let mut probe = path;
    while !probe.exists() {
        match probe.parent() {
            Some(parent) => probe = parent,
            None => return false,
        }
    }
    probe.canonicalize().is_ok_and(|p| p.starts_with(base))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_stay_inside() {
        assert_eq!(contained("sd_xl_base_1.0.safetensors"), Some(PathBuf::from("sd_xl_base_1.0.safetensors")));
        assert_eq!(contained("sdxl/style/x.safetensors"), Some(PathBuf::from("sdxl/style/x.safetensors")));
        assert_eq!(contained("portraits/2026-01-01-test"), Some(PathBuf::from("portraits/2026-01-01-test")));
        for bad in ["", ".", "/etc/passwd", "../x", "a/../../x", "a/..", "/home/user/Pictures/x"] {
            assert_eq!(contained(bad), None, "{bad:?} should be refused");
        }
    }

    #[test]
    fn below_follows_symlinks() {
        let root = std::env::temp_dir().join(format!("rustyui-paths-{}", std::process::id()));
        let (out, elsewhere) = (root.join("out"), root.join("elsewhere"));
        std::fs::create_dir_all(&out).unwrap();
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::os::unix::fs::symlink(&elsewhere, out.join("link")).unwrap();
        assert!(below(&out, &out.join("new/deeper/file.png")));
        assert!(!below(&out, &out.join("link/file.png")));
        assert!(!below(&out, &out.join("link/new/file.png")));
        assert!(!below(&out, &root.join("elsewhere/file.png")));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
