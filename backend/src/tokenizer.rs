//! CLIP's byte-level BPE, the way transformers' CLIPTokenizer runs it inside ComfyUI
//! (ftfy present: `whitespace_clean(fix_text(text)).lower()`, then the BPE pattern).

use anyhow::{Context, Result};
use regex::Regex;
use std::cell::RefCell;
use std::collections::HashMap;

pub const BOS: u32 = 49406;
pub const EOS: u32 = 49407;

static VOCAB: &str = include_str!("../assets/clip/vocab.json");
static MERGES: &str = include_str!("../assets/clip/merges.txt");

pub struct ClipTokenizer {
    encoder: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_enc: [char; 256],
    pat: Regex,
    cache: RefCell<HashMap<String, Vec<u32>>>,
}

/// GPT-2's reversible byte -> printable-character table.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    table
}

/// The parts of ftfy.fix_text that matter for prompts: curly quotes made straight.
fn fix_text(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            c => c,
        })
        .collect()
}

impl ClipTokenizer {
    pub fn new() -> Result<Self> {
        let encoder: HashMap<String, u32> = serde_json::from_str(VOCAB).context("CLIP vocab.json")?;
        // transformers: merges[1 : 49152 - 256 - 2 + 1]
        let mut ranks = HashMap::new();
        for (i, line) in MERGES.trim().lines().skip(1).take(49152 - 256 - 2).enumerate() {
            let mut parts = line.split(' ');
            if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                ranks.insert((a.to_string(), b.to_string()), i);
            }
        }
        let pat = Regex::new(r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")?;
        Ok(ClipTokenizer { encoder, ranks, byte_enc: bytes_to_unicode(), pat, cache: RefCell::new(HashMap::new()) })
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        if word.len() == 1 {
            return word;
        }
        loop {
            let best = word
                .windows(2)
                .filter_map(|p| self.ranks.get(&(p[0].clone(), p[1].clone())).map(|r| (*r, p[0].clone(), p[1].clone())))
                .min_by_key(|(r, _, _)| *r);
            let Some((_, first, second)) = best else { break };
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
            if word.len() == 1 {
                break;
            }
        }
        word
    }

    /// `tokenizer(text)["input_ids"][1:-1]`: the ids without BOS/EOS.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(hit) = self.cache.borrow().get(text) {
            return hit.clone();
        }
        let clean = fix_text(text).split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = vec![];
        for m in self.pat.find_iter(&clean) {
            let mapped: String = m.as_str().bytes().map(|b| self.byte_enc[b as usize]).collect();
            for piece in self.bpe(&mapped) {
                // an unknown piece is <|endoftext|> in CLIP's vocab, as transformers does
                ids.push(*self.encoder.get(&piece).unwrap_or(&EOS));
            }
        }
        self.cache.borrow_mut().insert(text.to_string(), ids.clone());
        ids
    }
}
