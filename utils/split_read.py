import os
from PIL import Image
import numpy as np

input_dir = "${HOME}/dev/python/formatted/IAM_page/test"
output_dir = "${HOME}/dev/python/formatted/IAM_224/test"
patch_size = 224

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert("RGB")
    img = img.resize((1216, 1760))
    W, H = img.size

    count = 0
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patch_np = np.array(patch.convert("L")) / 255.0

            ink_ratio = (patch_np < 0.85).mean()

            if ink_ratio < 0.05:
                continue  # skip mostly empty patches
            out_name = f"{os.path.splitext(fname)[0]}_{count:05d}.jpg"
            patch.save(os.path.join(output_dir, out_name))

            count += 1