import pandas as pd
import os
import cv2

test_csv = './Data/df_test.csv'
test_img_dir = './Data/test/'
out_dir = './Data/test_CLAHE/'

os.makedirs(out_dir, exist_ok=True)
df_test = pd.read_csv(test_csv)

missing = 0
for fname in df_test.dropna(subset=['filename'])['filename']:
    src_path = os.path.join(test_img_dir, fname)
    dst_path = os.path.join(out_dir, fname)
    if not os.path.exists(src_path):
        print(f"[Test] File not found: {src_path}")
        missing += 1
        continue
    img = cv2.imread(src_path, 0)
    if img is None:
        print(f"[Test] Could not read: {src_path}")
        missing += 1
        continue
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
    img_clahe = clahe.apply(img)
    cv2.imwrite(dst_path, img_clahe)
print(f"[Test] CLAHE applied and saved to {out_dir}. Missing/Unreadable: {missing}") 