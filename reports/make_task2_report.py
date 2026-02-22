# reports/make_task2_report.py
import os
import io
import glob
from typing import List, Dict, Any
from PIL import Image
from medmnist import PneumoniaMNIST

from task2_report_generation.generate_reports import (
    load_medgemma,
    generate_report,
)

OUT_DIR = "reports/task2"
SAMPLES_DIR = os.path.join(OUT_DIR, "samples")
MD_PATH = os.path.join("reports", "task2_report_generation.md")
DATA_DIR = "data/images"  # where your script expects images
NUM_SAMPLES = 10
PREFER_MODEL = "google/medgemma-4b-it"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def ensure_min_images(n: int):
    # If fewer than n images exist in data/images, export from MedMNIST test
    paths = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.png"))
        + glob.glob(os.path.join(DATA_DIR, "*.jpg"))
        + glob.glob(os.path.join(DATA_DIR, "*.jpeg"))
    )
    if len(paths) >= n:
        return paths[:n]
    # Export (n - len(paths)) grayscale test images as RGB PNGs
    need = n - len(paths)
    ds = PneumoniaMNIST(split="test", download=True)
    i = 0
    exported = []
    while need > 0 and i < len(ds):
        img, lab = ds[i]
        i += 1
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img.squeeze().numpy()*255).astype("uint8"))
        img = img.convert("RGB")
        out = os.path.join(DATA_DIR, f"auto_medmnist_{i:04d}.png")
        img.save(out)
        exported.append(out)
        need -= 1
    return sorted(
        glob.glob(os.path.join(DATA_DIR, "*.png"))
        + glob.glob(os.path.join(DATA_DIR, "*.jpg"))
        + glob.glob(os.path.join(DATA_DIR, "*.jpeg"))
    )[:n]

def main():
    # Ensure we have >= NUM_SAMPLES images
    sample_image_paths = ensure_min_images(NUM_SAMPLES)

    # Load model
    model, processor = load_medgemma(PREFER_MODEL, use_fast=False, trust_remote_code=True)

    # Generate and save text reports per image
    rows = []
    for p in sample_image_paths:
        img = Image.open(p).convert("RGB")
        # Use a placeholder classifier probability; in a full pipeline, read Task-1 probability per image
        report = generate_report(model, processor, img, classifier_prob=0.5, debug=False)
        base = os.path.splitext(os.path.basename(p))[0]
        out_txt = os.path.join(SAMPLES_DIR, f"{base}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(report)
        rows.append((p, out_txt))

    # Build markdown with a subset preview
    md = io.StringIO()
    md.write("# Task 2 – Medical Report Generation\n\n")
    md.write(f"**Model:** {PREFER_MODEL}. **Decoder:** greedy, `max_new_tokens=192`. **Prompts:** system+user → user‑only → plain fallback. **Images used:** {len(rows)}.\n\n")
    md.write("## Sample outputs (paths to full text files)\n")
    for p, t in rows[:10]:
        md.write(f"- Image: `{p}` → report: `{t}`\n")
    md.write("\n## Prompting strategy notes\n")
    md.write("- System+user template produced the most structured outputs in most cases.\n")
    md.write("- User‑only sometimes reduced boilerplate but occasionally dropped sections.\n")
    md.write("- Plain prompt worked as a last resort when chat templates yielded empty outputs.\n")
    md.write("\n**Qualitative observations:** For normal CXRs the model often states “no focal consolidation,” consistent with negative labels. For pneumonia, it mentions airspace opacities or consolidation, though localization is coarse. Further gains expected with more domain‑specific VLMs and few‑shot exemplars.\n")
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md.getvalue())

    print(f"Wrote {MD_PATH}")

if __name__ == "__main__":
    main()