import argparse
import os
import glob
import re
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
# DEVICE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# UTILITIES
# ============================================================
def print_debug(debug: bool, *args):
    if debug:
        print(*args)


def has_substantive_text(s: str) -> bool:
    """True if there's non-trivial alphanumeric content beyond our headings."""
    s_clean = re.sub(r"\s+", " ", s).strip().lower()
    s_clean = s_clean.replace("findings:", "").replace("impression:", "").replace("pneumonia present (yes/no):", "")
    return bool(re.search(r"[a-z0-9]", s_clean))


def normalize_report(text: str) -> str:
    """Remove role markers and enforce headings without inventing content."""
    lines = [ln.strip() for ln in text.splitlines()]
    bad_prefixes = ("user", "assistant", "model", "<start_of_turn>", "<end_of_turn>")
    lines = [ln for ln in lines if ln and not any(ln.lower().startswith(p) for p in bad_prefixes)]
    out = "\n".join(lines).strip()
    if "Findings:" not in out:
        out = "Findings:\n" + out
    if "Impression:" not in out:
        out += "\n\nImpression:\n"
    if "Pneumonia present (yes/no):" not in out:
        out += "\n\nPneumonia present (yes/no): "
    return out.strip()


def ensure_generation_prompt(chat_text: str) -> str:
    """
    Gemma-style templates should end with '<start_of_turn>model'.
    If not, add it. This avoids immediate EOS on some checkpoints.
    """
    tail = chat_text.rstrip()
    if not tail.endswith("<start_of_turn>model"):
        chat_text = tail + "\n<start_of_turn>model"
    return chat_text


def strip_prompt_if_present(output_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Works whether generate() returns full sequence (prompt + continuation),
    or continuation only. Strip prompt prefix only if present.
    """
    if output_ids is None or output_ids.numel() == 0:
        return output_ids

    B, T_out = output_ids.shape
    _, T_in = input_ids.shape
    if T_out <= T_in:
        return output_ids  # already continuation-only (or empty)

    try:
        pref_equal = torch.equal(output_ids[:, :T_in].to("cpu"), input_ids[:, :T_in].to("cpu"))
    except Exception:
        pref_equal = False

    return output_ids[:, T_in:] if pref_equal else output_ids


def decode_continuation(output_ids: torch.Tensor, input_ids: torch.Tensor, tokenizer) -> str:
    gen_ids = strip_prompt_if_present(output_ids, input_ids)
    text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text.replace("<start_of_turn>model", "").replace("<end_of_turn>", "").strip()


# ============================================================
# LOADERS
# ============================================================
def load_medgemma(model_name: str, use_fast: bool = False, trust_remote_code: bool = True):
    """
    Load model + processor in float32 for stability on T4.
    """
    print(f"Device: {DEVICE}")
    print(f"Loading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, use_fast=use_fast, trust_remote_code=trust_remote_code)

    # Force float32 to avoid FP16 NaNs / invalid probs on T4
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float32,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    # Make matmul as accurate as possible
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return model, processor


def load_samples(num_samples: int):
    image_dir = "data/images"
    if not os.path.exists(image_dir):
        raise RuntimeError("Directory data/images does not exist.")
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    if len(image_paths) == 0:
        raise RuntimeError("No images found in data/images.")
    samples = []
    for path in image_paths[:num_samples]:
        samples.append({"image_path": path, "classifier_prob": 0.99})
    return samples


# ============================================================
# PROMPTS
# ============================================================
def build_messages_system_user(classifier_prob: float) -> List[Dict[str, Any]]:
    system_text = (
        "You are a radiology assistant. Output a concise, structured report with EXACT headings:\n"
        "Findings:\n"
        "Impression:\n"
        "Pneumonia present (yes/no): yes|no\n"
        "Do not repeat the prompt. Do not include role tags."
    )
    user_text = (
        f"Analyze this chest X-ray carefully.\n"
        f"Classifier pneumonia probability: {classifier_prob:.3f}\n"
        f"Write a structured radiology report.\n"
        f"Findings:\n"
        f"Impression:\n"
        f"Pneumonia present (yes/no):"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]


def build_messages_user_only(classifier_prob: float) -> List[Dict[str, Any]]:
    user_text = (
        "You are a radiology assistant. Return ONLY these sections exactly, filled with concise clinical text:\n"
        "Findings:\n"
        "Impression:\n"
        "Pneumonia present (yes/no): yes|no\n\n"
        f"Analyze this chest X-ray.\n"
        f"Classifier pneumonia probability: {classifier_prob:.3f}\n"
        f"Now produce the report:"
    )
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]


def build_plain_prompt(classifier_prob: float) -> str:
    return (
        "You are a radiology assistant. Provide a structured report with EXACT headings:\n"
        "Findings:\n"
        "Impression:\n"
        "Pneumonia present (yes/no): yes|no\n\n"
        f"Task: Analyze this chest X-ray. Classifier pneumonia probability: {classifier_prob:.3f}\n"
        "Now output the report:"
    )


# ============================================================
# GENERATION (GREEDY ONLY, STABLE)
# ============================================================
def generate_chat_once(model, processor, image: Image.Image, messages: List[Dict[str, Any]], debug: bool) -> str:
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_text = ensure_generation_prompt(chat_text)

    inputs = processor(text=chat_text, images=[image], return_tensors="pt").to(model.device)

    # EOS/PAD: ensure we have valid IDs; set PAD to EOS if missing
    tok = processor.tokenizer
    eos_id = getattr(tok, "eos_token_id", None) or getattr(model.config, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None and hasattr(tok, "eos_token_id"):
        pad_id = tok.eos_token_id
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass

    print_debug(debug, f"[DEBUG] chat_text head: {chat_text[:160].replace(chr(10), ' | ')} ...")
    print_debug(debug, f"[DEBUG] input_ids shape: {tuple(inputs['input_ids'].shape)}; pixel batch: {inputs.get('pixel_values', torch.empty(0)).shape}")

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,          # GREEDY ONLY (stable on T4)
            num_beams=1,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
        )

    decoded = decode_continuation(out_ids, inputs["input_ids"], tok).strip()
    print_debug(debug, f"[DEBUG] decoded_len={len(decoded)}; head={decoded[:80]!r}")
    return decoded


def generate_plain_once(model, processor, image: Image.Image, prompt: str, debug: bool) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    tok = processor.tokenizer
    eos_id = getattr(tok, "eos_token_id", None) or getattr(model.config, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None and hasattr(tok, "eos_token_id"):
        pad_id = tok.eos_token_id
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass

    print_debug(debug, f"[DEBUG] plain_prompt head: {prompt[:160].replace(chr(10), ' | ')} ...")
    print_debug(debug, f"[DEBUG] input_ids shape: {tuple(inputs['input_ids'].shape)}; pixel batch: {inputs.get('pixel_values', torch.empty(0)).shape}")

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,          # GREEDY ONLY
            num_beams=1,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
        )

    decoded = decode_continuation(out_ids, inputs["input_ids"], tok).strip()
    print_debug(debug, f"[DEBUG] decoded_len={len(decoded)}; head={decoded[:80]!r}")
    return decoded


def generate_report(model, processor, image: Image.Image, classifier_prob: float, debug: bool) -> str:
    # 1) System+User chat
    decoded = generate_chat_once(model, processor, image, build_messages_system_user(classifier_prob), debug)
    norm = normalize_report(decoded)
    if has_substantive_text(norm):
        return norm

    # 2) User-only chat
    decoded = generate_chat_once(model, processor, image, build_messages_user_only(classifier_prob), debug)
    norm = normalize_report(decoded)
    if has_substantive_text(norm):
        return norm

    # 3) Plain multimodal prompt (no chat template)
    decoded = generate_plain_once(model, processor, image, build_plain_prompt(classifier_prob), debug)
    norm = normalize_report(decoded)
    if has_substantive_text(norm):
        return norm

    fail_note = "[Generation note: model returned empty on greedy decoding across chat (system+user), chat (user-only), and plain prompts.]"
    base = "Findings:\n\nImpression:\n\nPneumonia present (yes/no): "
    return f"{base}\n\n{fail_note}"


# ============================================================
# PIPELINE / CLI
# ============================================================
def run(num_samples: int, prefer_model: str, use_fast: bool, debug: bool):
    model, processor = load_medgemma(prefer_model, use_fast=use_fast, trust_remote_code=True)
    samples = load_samples(num_samples)

    print("\n===== GENERATING REPORTS =====\n")
    for s in samples:
        image = Image.open(s["image_path"]).convert("RGB")
        report = generate_report(
            model=model,
            processor=processor,
            image=image,
            classifier_prob=s["classifier_prob"],
            debug=debug,
        )
        print("=" * 60)
        print(f"Image: {os.path.basename(s['image_path'])}\n")
        print(report if report else "[EMPTY OUTPUT]")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--prefer_model", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--use_fast", action="store_true", help="Use the fast image processor (default: slow).")
    parser.add_argument("--debug", action="store_true", help="Verbose internal logging to diagnose empty outputs.")
    args = parser.parse_args()

    run(
        num_samples=args.num_samples,
        prefer_model=args.prefer_model,
        use_fast=args.use_fast,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()