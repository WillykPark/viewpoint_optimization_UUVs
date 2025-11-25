import json
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("/blue/eel6825/yo.park/APRIL/PPO/dataset/Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped")
IMG  = ROOT

EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG"]
EXCLUDE = {"rotating-platform", ".DS_Store"}

classes = sorted([d.name for d in IMG.iterdir() if d.is_dir() and d.name not in EXCLUDE])
label_map = {c: i for i, c in enumerate(classes)}   #make a dictionary

# (path, labels) pair
pairs = []
for c in classes:
    for ext in EXTS:
        for p in (IMG / c).rglob(f"*{ext}"):
            pairs.append((str(p), label_map[c]))

if not pairs:
    raise RuntimeError(f"No images found under {IMG}. Check EXTS / path.")

# Stratified split: train/test -> train/val
y = [lbl for _, lbl in pairs]
train, test = train_test_split(pairs, test_size=0.15, stratify=y, random_state=42)

y_tr = [lbl for _, lbl in train]
val_ratio_in_train = 0.15 / (1.0 - 0.15)   # 전체 15%가 val이 되도록
train, val = train_test_split(train, test_size=val_ratio_in_train, stratify=y_tr, random_state=42)


S = ROOT / "splits"
S.mkdir(parents=True, exist_ok=True)

def dump(lst, path):
    with open(path, "w") as f:
        for p, l in lst:
            f.write(f"{p} {l}\n")

dump(train, S / "train.txt")
dump(val,   S / "val.txt")
dump(test,  S / "test.txt")

with open(S / "classes.json", "w") as f:
    json.dump(label_map, f, indent=2, ensure_ascii=False)

print("Saved splits to", S)
print("Classes:", label_map)