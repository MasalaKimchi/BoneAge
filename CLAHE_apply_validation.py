import pandas as pd
import os
import cv2

val_csv = './Data/df_val.csv'
val_img_dir = './Data/validation/'
out_dir = './Data/validation_CLAHE/'

os.makedirs(out_dir, exist_ok=True)
df_val = pd.read_csv(val_csv)

missing = 0
for fname in df_val.dropna(subset=['filename'])['filename']:
    src_path = os.path.join(val_img_dir, fname)
    dst_path = os.path.join(out_dir, fname)
    if not os.path.exists(src_path):
        print(f"[Validation] File not found: {src_path}")
        missing += 1
        continue
    img = cv2.imread(src_path, 0)
    if img is None:
        print(f"[Validation] Could not read: {src_path}")
        missing += 1
        continue
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
    img_clahe = clahe.apply(img)
    cv2.imwrite(dst_path, img_clahe)
print(f"[Validation] CLAHE applied and saved to {out_dir}. Missing/Unreadable: {missing}") 