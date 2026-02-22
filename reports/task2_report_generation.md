# Task 2 – Medical Report Generation

**Model:** google/medgemma-4b-it. **Decoder:** greedy, `max_new_tokens=192`. **Prompts:** system+user → user‑only → plain fallback. **Images used:** 10.

## Sample outputs (paths to full text files)
- Image: `data/images/auto_medmnist_0001.png` → report: `reports/task2/samples/auto_medmnist_0001.txt`
- Image: `data/images/auto_medmnist_0002.png` → report: `reports/task2/samples/auto_medmnist_0002.txt`
- Image: `data/images/auto_medmnist_0003.png` → report: `reports/task2/samples/auto_medmnist_0003.txt`
- Image: `data/images/auto_medmnist_0004.png` → report: `reports/task2/samples/auto_medmnist_0004.txt`
- Image: `data/images/auto_medmnist_0005.png` → report: `reports/task2/samples/auto_medmnist_0005.txt`
- Image: `data/images/auto_medmnist_0006.png` → report: `reports/task2/samples/auto_medmnist_0006.txt`
- Image: `data/images/auto_medmnist_0007.png` → report: `reports/task2/samples/auto_medmnist_0007.txt`
- Image: `data/images/img1.png` → report: `reports/task2/samples/img1.txt`
- Image: `data/images/img2.png` → report: `reports/task2/samples/img2.txt`
- Image: `data/images/img3.png` → report: `reports/task2/samples/img3.txt`

## Prompting strategy notes
- System+user template produced the most structured outputs in most cases.
- User‑only sometimes reduced boilerplate but occasionally dropped sections.
- Plain prompt worked as a last resort when chat templates yielded empty outputs.

**Qualitative observations:** For normal CXRs the model often states “no focal consolidation,” consistent with negative labels. For pneumonia, it mentions airspace opacities or consolidation, though localization is coarse. Further gains expected with more domain‑specific VLMs and few‑shot exemplars.
