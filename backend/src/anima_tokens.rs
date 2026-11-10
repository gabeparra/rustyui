//! Anima's prompt tokenization, as ComfyUI's AnimaTokenizer runs it (comfy/text_encoders/anima.py):
//! the prompt goes through sd1_clip.SDTokenizer twice, into Qwen3's BPE ids (the weights set to 1.0;
//! Qwen3-0.6B encodes these) and into T5's Unigram ids with their weights (Anima's LLM adapter
//! embeds those and cross-attends from them into Qwen3's output).
//!
//! The two pipelines are the ones transformers 5 builds in memory, which for T5 is not the
//! tokenizer.json beside it: refs/anima_tokens_ref.py writes them out as tokenizers JSON
//! (assets/anima/), and the same tokenizers release the Python package wraps (0.23.1, Oniguruma
//! regexes) runs them here.

use crate::prompt::{escape_important, split_embeddings, token_weights, try_embedding, unescape_important, Embeddings};
use anyhow::{bail, Result};
use tokenizers::Tokenizer;

static QWEN: &str = include_str!("../assets/anima/qwen3_06b_tokenizer.json");
static T5: &str = include_str!("../assets/anima/t5xxl_tokenizer.json");

/// Qwen3Tokenizer's pad_token (<|endoftext|>): an empty prompt is this one token (min_length 1)
pub const QWEN_PAD: u32 = 151643;
/// </s>, which T5's post-processor appends and SDTokenizer moves to the end of the whole prompt
pub const T5_END: u32 = 1;

pub struct AnimaTokenizer {
    qwen: Tokenizer,
    t5: Tokenizer,
}

/// One prompt's tokens: Qwen3's ids, and T5's ids with their prompt weights (Python floats;
/// ComfyUI makes a float32 tensor of them)
pub struct Tokens {
    pub qwen: Vec<u32>,
    pub t5: Vec<(u32, f64)>,
}

impl AnimaTokenizer {
    pub fn new() -> Result<Self> {
        let load = |json: &str| json.parse::<Tokenizer>().map_err(|e| anyhow::anyhow!("Anima tokenizer: {e}"));
        Ok(AnimaTokenizer { qwen: load(QWEN)?, t5: load(T5)? })
    }

    pub fn tokenize(&self, text: &str, embs: &Embeddings) -> Result<Tokens> {
        let qwen: Vec<u32> = sd_tokenize(&self.qwen, false, text, embs)?.into_iter().map(|(id, _)| id).collect();
        let mut t5 = sd_tokenize(&self.t5, true, text, embs)?;
        t5.push((T5_END, 1.0));
        Ok(Tokens { qwen: if qwen.is_empty() { vec![QWEN_PAD] } else { qwen }, t5 })
    }
}

/// SDTokenizer.tokenize_with_weights up to the batching, which for both of Anima's tokenizers
/// (no start token, no length limit, no padding but to one token) only appends T5's end token:
/// each weighted segment through the pipeline, with its end token cut when the pipeline adds one
fn sd_tokenize(tok: &Tokenizer, adds_end: bool, text: &str, embs: &Embeddings) -> Result<Vec<(u32, f64)>> {
    let mut out = vec![];
    for (segment, weight) in token_weights(&escape_important(text), 1.0) {
        for word in split_embeddings(&unescape_important(&segment)) {
            let mut word = word;
            if let Some(raw) = word.strip_prefix("embedding:") {
                // any file of that name is an embedding to ComfyUI, which takes its first tensor
                // when it has none for Qwen3 or T5
                let (found, name, leftover) = try_embedding(raw.trim_matches('\n'), |n| embs.find(n));
                if let Some(path) = found {
                    bail!("embedding:{name} ({}): embeddings in an Anima prompt are not supported yet", path.display());
                }
                eprintln!("[engine] warning, embedding:{name} does not exist, ignoring");
                if leftover.is_empty() {
                    continue;
                }
                word = leftover;
            }
            let enc = tok.encode(word.as_str(), true).map_err(|e| anyhow::anyhow!("tokenizing {word:?}: {e}"))?;
            let ids = enc.get_ids();
            let ids = if adds_end { &ids[..ids.len().saturating_sub(1)] } else { ids };
            out.extend(ids.iter().map(|&id| (id, weight)));
        }
    }
    Ok(out)
}

/// debug-anima-tokens REF.json (refs/anima_tokens_ref.py): every prompt in it tokenized here and
/// compared with ComfyUI's lists, id for id and weight for weight
pub fn debug_tokens(reference: &std::path::Path, embs: &Embeddings) -> Result<()> {
    let cases: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(reference)?)?;
    let tok = AnimaTokenizer::new()?;
    let mut bad = 0;
    for case in &cases {
        let text = case["text"].as_str().unwrap_or_default();
        let want = |key: &str| -> Vec<(u32, f64)> {
            case[key].as_array().into_iter().flatten().map(|p| (p[0].as_u64().unwrap() as u32, p[1].as_f64().unwrap())).collect()
        };
        let got = tok.tokenize(text, embs)?;
        let got_qwen: Vec<(u32, f64)> = got.qwen.iter().map(|&id| (id, 1.0)).collect();
        let short: String = text.chars().take(48).collect::<String>().replace('\n', "\\n");
        for (name, w, g) in [("qwen3_06b", want("qwen3_06b"), got_qwen), ("t5xxl", want("t5xxl"), got.t5)] {
            if w == g {
                continue;
            }
            bad += 1;
            let at = w.iter().zip(&g).position(|(a, b)| a != b).unwrap_or(w.len().min(g.len()));
            println!(
                "{name} {short:?}: {} tokens here, {} in ComfyUI; first difference at {at}: {:?} against {:?}",
                g.len(),
                w.len(),
                g.get(at),
                w.get(at)
            );
        }
    }
    println!("{} prompts, {bad} token lists differ{}", cases.len(), if bad == 0 { " (all identical)" } else { "" });
    if bad > 0 {
        bail!("{bad} token lists differ");
    }
    Ok(())
}
