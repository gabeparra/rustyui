//! The prompt writer: a short idea in, a whole SDXL prompt (comma-separated tags) out, written by an
//! LLM inside the backend, on the card.
//!
//! The model is any GGUF Gemma 3 instruct model, run by a copy of candle's quantized Gemma 3 (`qgemma3`;
//! llama.cpp's formats, so a Q4_K_M or Q6_K file is used as it is). The tokenizer is Gemma 3's SentencePiece
//! model, read from the LTX-2.3 Gemma text-encoder file that carries it ("spiece_model"); Gemma 3
//! fine-tunes keep Gemma's tokenizer, so the same file serves them all.
//!
//! The chat is Gemma's template, assembled from ids as Hugging Face's tokenizer would: the
//! special tokens by id, the text between them by SentencePiece.

use crate::qgemma3::ModelWeights;
use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;
use sentencepiece::SentencePieceProcessor;
use std::path::Path;
use std::time::Instant;

/// Gemma has no system turn: its instructions open the user's first turn. One worked example
/// says more about the shape of an answer than a list of categories, which a 12B model echoes back
/// when it is told them in the abstract.
const INSTRUCTIONS: &str = "\
You write prompts for an SDXL image model trained on comma-separated descriptive tags. Turn the request into one \
line of 25 to 40 comma-separated tags, most important first: the quality tags, then the subject \
(who or what, how many, colours, build), then what is happening (pose, action, expression, \
clothing), then where and how (setting, props, light, framing).

Rules: real tags only, lowercase, no sentences, no repeated tags. When a current prompt is given, \
keep its quality tags, its style tags and trigger words (which may look like names), and its \
subject, and rewrite the rest to fit the request.

Example request: an old fisherman mending nets on a pier at dawn
Example answer: masterpiece, best quality, highly detailed, one man, solo, old man, grey beard, \
wrinkled skin, knit cap, wool sweater, sitting, mending net, fishing net, focused, wooden pier, \
boats, harbor, ocean, dawn, mist, soft lighting, outdoors, full body, from side

Example request: a lighthouse in a storm
Example answer: masterpiece, best quality, highly detailed, no humans, scenery, lighthouse, rocky \
shore, storm, dark clouds, lightning, crashing waves, rain, light beam, night, dramatic lighting, \
wide shot

Answer with the tags only, on one line.";

pub struct Writer {
    model: ModelWeights,
    sp: SentencePieceProcessor,
    device: Device,
    bos: u32,
    start_turn: u32,
    end_turn: u32,
    eos: u32,
}

/// The tokenizer file below a ComfyUI models folder: the Gemma 3 text encoder LTX-2.3 uses,
/// which carries the SentencePiece model.
pub fn default_tokenizer(models: &Path) -> std::path::PathBuf {
    models.join("text_encoders/gemma_3_12B_it_fp4_mixed.safetensors")
}

/// What one request wrote, and how long the parts took.
pub struct Written {
    pub prompt: String,
    pub tokens: usize,
    pub seconds: f64,
}

fn piece(sp: &SentencePieceProcessor, p: &str) -> Result<u32> {
    sp.piece_to_id(p)?.with_context(|| format!("the tokenizer has no {p}"))
}

impl Writer {
    /// `gguf`: the model; `tokenizer`: a safetensors file holding Gemma 3's "spiece_model".
    pub fn load(gguf: &Path, tokenizer: &Path, device: &Device) -> Result<Self> {
        let t = Instant::now();
        let st = unsafe { MmapedSafetensors::new(tokenizer) }.with_context(|| format!("open {}", tokenizer.display()))?;
        let proto = st.load("spiece_model", &Device::Cpu)?.to_vec1::<u8>()?;
        let sp = SentencePieceProcessor::from_serialized_proto(&proto).context("Gemma SentencePiece model")?;
        let mut file = std::fs::File::open(gguf).with_context(|| format!("open {}", gguf.display()))?;
        let content = gguf_file::Content::read(&mut file).with_context(|| format!("read {} as GGUF", gguf.display()))?;
        let arch = content.metadata.get("general.architecture").and_then(|v| v.to_string().ok()).cloned().unwrap_or_default();
        if arch != "gemma3" {
            bail!("{} is a {arch:?} model; the writer runs Gemma 3", gguf.display());
        }
        let model = ModelWeights::from_gguf(content, &mut file, device)?;
        let w = Writer {
            bos: piece(&sp, "<bos>")?,
            start_turn: piece(&sp, "<start_of_turn>")?,
            end_turn: piece(&sp, "<end_of_turn>")?,
            eos: piece(&sp, "<eos>")?,
            model,
            sp,
            device: device.clone(),
        };
        eprintln!("[rustyui] prompt writer {} loaded in {:.1}s", gguf.display(), t.elapsed().as_secs_f64());
        Ok(w)
    }

    /// A token with no letter or digit in it: a comma, a space, a full stop.
    fn is_separator(&self, id: u32) -> bool {
        self.sp.decode_piece_ids(&[id]).is_ok_and(|t| !t.chars().any(char::is_alphanumeric))
    }

    fn text(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self.sp.encode(s).context("SentencePiece encode")?.into_iter().map(|p| p.id).collect())
    }

    /// Gemma's chat template for one user turn, as ids.
    fn chat(&self, user: &str) -> Result<Vec<u32>> {
        let mut ids = vec![self.bos, self.start_turn];
        ids.extend(self.text(&format!("user\n{user}"))?);
        ids.push(self.end_turn);
        ids.extend(self.text("\n")?);
        ids.push(self.start_turn);
        ids.extend(self.text("model\n")?);
        Ok(ids)
    }

    /// The prompt for `idea`, building on `current` when it is not empty.
    pub fn write(&mut self, idea: &str, current: &str, seed: u64) -> Result<Written> {
        let t = Instant::now();
        let mut user = format!("{INSTRUCTIONS}\n\nRequest: {}", idea.trim());
        if !current.trim().is_empty() {
            user.push_str(&format!("\nCurrent prompt: {}", current.trim()));
        }
        let ids = self.chat(&user)?;
        let mut sampler = LogitsProcessor::from_sampling(seed, Sampling::TopKThenTopP { k: 64, p: 0.95, temperature: 0.8 });
        let mut logits = self.model.forward(&Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?, 0)?.squeeze(0)?;
        let mut pos = ids.len();
        let mut out: Vec<u32> = vec![];
        let mut commas = 0;
        for _ in 0..MAX_TOKENS {
            // llama.cpp's repeat penalty over what was just written: a tag list otherwise falls
            // into a loop of the same few words until the token budget runs out. The commas
            // between tags are left out of it: a list repeats them by design, and penalising
            // them made the model end its answer after about a dozen tags
            let recent: Vec<u32> =
                out[out.len().saturating_sub(REPEAT_WINDOW)..].iter().copied().filter(|&id| !self.is_separator(id)).collect();
            let scored = if recent.is_empty() { logits.clone() } else { apply_repeat_penalty(&logits, REPEAT_PENALTY, &recent)? };
            let next = sampler.sample(&scored)?;
            if next == self.end_turn || next == self.eos {
                break;
            }
            out.push(next);
            commas += self.sp.decode_piece_ids(&[next]).map_or(0, |t| t.matches(',').count());
            if commas >= MAX_TAGS {
                break;
            }
            // the answer is one line: a line break after some tags ends it
            if self.sp.decode_piece_ids(&[next]).is_ok_and(|t| t.contains('\n')) && out.len() > 8 {
                break;
            }
            logits = self.model.forward(&Tensor::new(&[next], &self.device)?.unsqueeze(0)?, pos)?.squeeze(0)?;
            pos += 1;
        }
        let text = self.sp.decode_piece_ids(&out).context("SentencePiece decode")?;
        Ok(Written { prompt: tidy(&text), tokens: out.len(), seconds: t.elapsed().as_secs_f64() })
    }
}

const MAX_TOKENS: usize = 160;
/// The instructions ask for 25 to 40 tags; a list that reaches 40 is cut there.
const MAX_TAGS: usize = 40;
const REPEAT_PENALTY: f32 = 1.15;
const REPEAT_WINDOW: usize = 64;

/// The answer's first line of tags: no quotes, no closing full stop, single spaces, and each tag
/// once (the first time it came).
fn tidy(text: &str) -> String {
    let line = text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let line = line.trim_matches(|c| c == '"' || c == '`').trim_end_matches('.').trim();
    let mut seen = std::collections::HashSet::new();
    line.split(',')
        .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|t| !t.is_empty() && seen.insert(t.to_lowercase()))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_are_tidied_to_one_line_of_tags() {
        assert_eq!(tidy("\n  \"masterpiece,  best quality, solo.\"\nHere you go"), "masterpiece, best quality, solo");
        assert_eq!(tidy("`solo, scenery`"), "solo, scenery");
        assert_eq!(tidy(""), "");
        assert_eq!(tidy("calm, cool, calm, Calm, cool,, heroic"), "calm, cool, heroic");
    }
}
