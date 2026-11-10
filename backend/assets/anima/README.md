# Anima's tokenizers

`qwen3_06b_tokenizer.json` and `t5xxl_tokenizer.json` are the tokenizer pipelines ComfyUI's
`AnimaTokenizer` runs (`comfy/text_encoders/anima.py`), serialized by `tokenizers` 0.23.1 from the
objects transformers 5.16.1 builds in memory: `backend/refs/anima_tokens_ref.py assets` writes them.
They are made from ComfyUI's `qwen25_tokenizer` (Qwen's vocabulary and merges, Apache-2.0) and
`t5_tokenizer` (Google's T5 vocabulary, Apache-2.0). T5's is not the `tokenizer.json` ComfyUI ships:
transformers 5 replaces its pre-tokenizer with WhitespaceSplit and Metaspace "always".
