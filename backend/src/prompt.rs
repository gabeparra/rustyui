//! ComfyUI's prompt language, ported from comfy/sd1_clip.py (tokenize_with_weights).
//!
//! `(word:1.2)` and `(word)` weights, `\(` escapes, `embedding:name` (only after whitespace or
//! at the start of a weighted piece, as ComfyUI splits on `(?<=\s)embedding:`), and the
//! packing into 77-token chunks with ComfyUI's rule that a group shorter than 8 tokens is
//! never split across chunks.

use crate::tokenizer::{ClipTokenizer, BOS, EOS};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum Tok {
    Id(u32),
    /// One vector of a textual-inversion embedding, for this encoder's width.
    Vec(Vec<f32>),
}

pub type Chunk = Vec<(Tok, f32)>;

pub fn escape_important(text: &str) -> String {
    text.replace("\\)", "\0\u{1}").replace("\\(", "\0\u{2}")
}

pub fn unescape_important(text: &str) -> String {
    text.replace("\0\u{1}", ")").replace("\0\u{2}", "(")
}

pub fn parse_parentheses(s: &str) -> Vec<String> {
    let mut result = vec![];
    let mut current = String::new();
    let mut level: i64 = 0;
    for c in s.chars() {
        if c == '(' {
            if level == 0 {
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                current = "(".into();
            } else {
                current.push(c);
            }
            level += 1;
        } else if c == ')' {
            level -= 1;
            if level == 0 {
                current.push(')');
                result.push(std::mem::take(&mut current));
            } else {
                current.push(c);
            }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// Python's float() on the text after the last ':'.
fn py_float(s: &str) -> Option<f64> {
    let t = s.trim().replace('_', "");
    match t.to_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => Some(f64::NAN),
        _ => t.parse::<f64>().ok(),
    }
}

pub fn token_weights(s: &str, current: f64) -> Vec<(String, f64)> {
    let mut out = vec![];
    for x in parse_parentheses(s) {
        let chars: Vec<char> = x.chars().collect();
        if chars.len() >= 2 && chars[0] == '(' && chars[chars.len() - 1] == ')' {
            let mut inner: String = chars[1..chars.len() - 1].iter().collect();
            let mut weight = current * 1.1;
            if let Some(pos) = inner.rfind(':') {
                if pos > 0 {
                    if let Some(w) = py_float(&inner[pos + 1..]) {
                        weight = w;
                        inner.truncate(pos);
                    }
                }
            }
            out.extend(token_weights(&inner, weight));
        } else {
            out.push((x, current));
        }
    }
    out
}

/// Splits like `re.split(r'(?<=\s)embedding:', text)`, then puts "embedding:" back on each
/// piece after the first.
pub fn split_embeddings(text: &str) -> Vec<String> {
    const ID: &str = "embedding:";
    let mut pieces = vec![];
    let mut start = 0;
    let mut search = 0;
    while let Some(rel) = text[search..].find(ID) {
        let at = search + rel;
        let prev_ws = text[..at].chars().next_back().is_some_and(char::is_whitespace);
        if prev_ws {
            pieces.push(text[start..at].to_string());
            start = at + ID.len();
        }
        search = at + ID.len();
    }
    pieces.push(text[start..].to_string());
    let mut out = vec![pieces[0].clone()];
    out.extend(pieces[1..].iter().map(|p| format!("{ID}{p}")));
    out.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Loads `name` from the embedding folders the way ComfyUI's load_embed does: exact file name,
/// or with .safetensors / .pt / .bin appended (case-sensitive), then the tensor under `key`.
pub struct Embeddings {
    pub dirs: Vec<PathBuf>,
    cache: HashMap<(String, String), Option<Vec<Vec<f32>>>>,
}

impl Embeddings {
    pub fn new(dirs: Vec<PathBuf>) -> Self {
        Embeddings { dirs, cache: HashMap::new() }
    }

    pub fn load(&mut self, name: &str, key: &str) -> Option<Vec<Vec<f32>>> {
        let ck = (name.to_string(), key.to_string());
        if let Some(hit) = self.cache.get(&ck) {
            return hit.clone();
        }
        let found = self.find(name).and_then(|path| read_embedding(&path, key));
        self.cache.insert(ck, found.clone());
        found
    }

    pub fn find(&self, name: &str) -> Option<PathBuf> {
        for dir in &self.dirs {
            let p = dir.join(name);
            // ComfyUI refuses a name that escapes its folder
            if !p.starts_with(dir) || name.contains("..") {
                continue;
            }
            if p.is_file() {
                return Some(p);
            }
            for ext in [".safetensors", ".pt", ".bin"] {
                let q = PathBuf::from(format!("{}{ext}", p.display()));
                if q.is_file() {
                    return Some(q);
                }
            }
        }
        None
    }
}

fn read_embedding(path: &PathBuf, key: &str) -> Option<Vec<Vec<f32>>> {
    if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        eprintln!("[engine] embedding {}: only .safetensors embeddings are read so far", path.display());
        return None;
    }
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(path).ok()? };
    let t = st.load(key, &candle_core::Device::Cpu).ok()?;
    let t = t.to_dtype(candle_core::DType::F32).ok()?;
    let t = if t.rank() == 1 { t.unsqueeze(0).ok()? } else { t };
    t.to_vec2::<f32>().ok()
}

pub struct Encoder<'a> {
    pub tok: &'a ClipTokenizer,
    pub pad: u32,
    /// "clip_l" or "clip_g": which tensor of an SDXL embedding file this encoder takes
    pub embedding_key: &'static str,
    pub max_length: usize,
}

/// SDTokenizer._try_get_embedding on the text after "embedding:": (what `load` found for the
/// name, the name, the leftover text to tokenize)
pub fn try_embedding<T>(raw: &str, mut load: impl FnMut(&str) -> Option<T>) -> (Option<T>, String, String) {
    let split: Vec<&str> = raw.split_whitespace().collect();
    let mut name = split.first().copied().unwrap_or("").to_string();
    let mut leftover = split.get(1..).map(|s| s.join(" ")).unwrap_or_default();
    if let Some(pos) = name.find(['<', '[']) {
        let tail = name[pos..].to_string();
        leftover = if leftover.is_empty() { tail } else { format!("{tail} {leftover}") };
        name.truncate(pos);
    }
    let mut found = load(&name);
    if found.is_none() {
        let stripped = name.trim_matches(',');
        if stripped.len() < name.len() {
            found = load(stripped);
            // ComfyUI's own slice: name[len(stripped):], which assumes the commas trail
            let rest = name.get(stripped.len()..).unwrap_or("").to_string();
            return (found, name, format!("{rest} {leftover}"));
        }
    }
    (found, name, leftover)
}

impl Encoder<'_> {
    /// _try_get_embedding: (vectors, leftover text)
    fn try_embedding(&self, embs: &mut Embeddings, raw: &str) -> (Option<Vec<Vec<f32>>>, String, String) {
        try_embedding(raw, |name| embs.load(name, self.embedding_key))
    }

    pub fn tokenize(&self, text: &str, embs: &mut Embeddings) -> Vec<Chunk> {
        let text = escape_important(text);
        let mut groups: Vec<Vec<(Tok, f32)>> = vec![];
        for (segment, weight) in token_weights(&text, 1.0) {
            let w = weight as f32;
            let segment = unescape_important(&segment);
            for word in split_embeddings(&segment) {
                let mut word = word;
                if let Some(raw) = word.strip_prefix("embedding:") {
                    let raw = raw.trim_matches('\n').to_string();
                    let (vectors, name, leftover) = self.try_embedding(embs, &raw);
                    match vectors {
                        Some(v) => groups.push(v.into_iter().map(|vec| (Tok::Vec(vec), w)).collect()),
                        None => eprintln!("[engine] warning, embedding:{name} does not exist, ignoring"),
                    }
                    if leftover.is_empty() {
                        continue;
                    }
                    word = leftover;
                }
                groups.push(self.tok.encode(&word).into_iter().map(|id| (Tok::Id(id), w)).collect());
            }
        }

        // reshape into 77-token chunks, exactly as ComfyUI does
        let max = self.max_length;
        let mut chunks: Vec<Chunk> = vec![];
        let mut batch: Chunk = vec![(Tok::Id(BOS), 1.0)];
        for group in groups {
            let is_large = group.len() >= 8;
            let mut group = group;
            while !group.is_empty() {
                if group.len() + batch.len() > max - 1 {
                    let remaining = max - batch.len() - 1;
                    if is_large {
                        let rest = group.split_off(remaining);
                        batch.extend(group);
                        batch.push((Tok::Id(EOS), 1.0));
                        group = rest;
                    } else {
                        batch.push((Tok::Id(EOS), 1.0));
                        batch.extend(std::iter::repeat_n((Tok::Id(self.pad), 1.0), remaining));
                    }
                    chunks.push(std::mem::replace(&mut batch, vec![(Tok::Id(BOS), 1.0)]));
                } else {
                    batch.extend(group);
                    group = vec![];
                }
            }
        }
        batch.push((Tok::Id(EOS), 1.0));
        while batch.len() < max {
            batch.push((Tok::Id(self.pad), 1.0));
        }
        chunks.push(batch);
        chunks
    }
}
