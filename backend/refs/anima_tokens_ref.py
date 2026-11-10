"""backend/refs/anima_tokens_ref.py: Anima's prompt tokenization as ComfyUI runs it (AnimaTokenizer:
Qwen3's BPE and T5's Unigram, each through sd1_clip.SDTokenizer's weights and padding). CPU only.

  anima_tokens_ref.py assets DIR             the two pipelines as transformers 5 builds them in memory,
                                             written as tokenizers JSON (backend/assets/anima/)
  anima_tokens_ref.py OUT.json [GRAPH...]    every prompt below, and each CLIPTextEncode text in the
                                             graphs, with the (id, weight) lists ComfyUI makes of it

transformers 5 builds T5's pipeline in code (WhitespaceSplit, Metaspace "always") rather than taking
the tokenizer.json beside it ("first"), so the files on disk are not what runs: the assets are the
live objects. Run in ComfyUI's folder with its venv:
  cd "$COMFYUI_DIR" && .venv/bin/python /path/to/rustyui/backend/refs/anima_tokens_ref.py OUT.json"""
import json, os, sys

sys.path.insert(0, os.getcwd())
import folder_paths
from comfy.text_encoders.anima import AnimaTokenizer

# what each piece of SDTokenizer and the two pipelines does with odd input
PROMPTS = [
    "",
    "a lighthouse on a cliff at dusk",
    "masterpiece,best quality,score_8,score_7, one cat, two dogs,  solo ,",
    "(red fox:1.3), (blue sky), [old], ((very detailed)), (a (nested:1.2) thing:0.8)",
    "escaped \\(parens\\) and a \\\\ backslash, (weighted \\(inner\\):1.1)",
    "embedding:nothing_here, then words",
    "Don't, it's, WE'LL, they'RE, I'M, you'd, s'mores",
    "line one\nline two\r\nline three\n\n  indented\ttabbed",
    "café crème, naïve, ＡＢＣ fullwidth, Ⅳ roman, ½ half",
    "猫と犬, 日本語の文, 한국어, русский",
    "emoji \U0001f43a\U0001f98a✨ and ❤️",
    "“curly quotes” and ‘single’, em—dash, ellipsis…",
    "<|endoftext|> <|im_start|>user</s> <pad> <extra_id_0> <unk>",
    "numbers 123456 3.14 1,000,000 and 8k uhd",
    "   leading and trailing spaces   ",
    "a" * 300,
    ", ".join(f"tag{i}" for i in range(400)),
]


def texts(graph_path):
    g = json.load(open(graph_path))
    for node in g.values():
        if node.get("class_type") == "CLIPTextEncode" and isinstance(node["inputs"].get("text"), str):
            yield node["inputs"]["text"]


def main():
    tok = AnimaTokenizer(embedding_directory=folder_paths.get_folder_paths("embeddings"))
    if sys.argv[1] == "assets":
        os.makedirs(sys.argv[2], exist_ok=True)
        for name, st in [("qwen3_06b", tok.qwen3_06b), ("t5xxl", tok.t5xxl)]:
            path = os.path.join(sys.argv[2], f"{name}_tokenizer.json")
            with open(path, "w") as f:
                f.write(st.tokenizer._tokenizer.to_str(pretty=False))
            print(path)
        return
    prompts = list(PROMPTS)
    for g in sys.argv[2:]:
        prompts += list(texts(g))
    out = []
    for text in prompts:
        t = tok.tokenize_with_weights(text)
        assert len(t["qwen3_06b"]) == 1 and len(t["t5xxl"]) == 1
        out.append({
            "text": text,
            "qwen3_06b": [[int(i), float(w)] for i, w in t["qwen3_06b"][0]],
            "t5xxl": [[int(i), float(w)] for i, w in t["t5xxl"][0]],
        })
    json.dump(out, open(sys.argv[1], "w"))
    print(f"{len(out)} prompts -> {sys.argv[1]}")


main()
