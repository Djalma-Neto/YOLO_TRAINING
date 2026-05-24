import os
import random
import numpy as np

from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

# =========================================================
# CONFIG
# =========================================================

DIR_AMOSTRAS = "./amostras"
DIR_OBJETOS = "./alvo"
BASE_DIR = "./dataset"
DIR_TRAIN_IMAGES = f"{BASE_DIR}/train/images"
DIR_TRAIN_LABELS = f"{BASE_DIR}/train/labels"
DIR_VAL_IMAGES = f"{BASE_DIR}/val/images"
DIR_VAL_LABELS = f"{BASE_DIR}/val/labels"

os.makedirs(DIR_AMOSTRAS, exist_ok=True)
os.makedirs(DIR_OBJETOS, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DIR_TRAIN_IMAGES, exist_ok=True)
os.makedirs(DIR_TRAIN_LABELS, exist_ok=True)
os.makedirs(DIR_VAL_IMAGES, exist_ok=True)
os.makedirs(DIR_VAL_LABELS, exist_ok=True)

TOTAL = 500
TRAIN_SPLIT = 0.8
NEGATIVE_RATIO = 0.15

CLASS_ID = 0
CLASS_NAME = "objeto"

# =========================================================
# CREATE DIRS
# =========================================================

dirs = [DIR_TRAIN_IMAGES,DIR_TRAIN_LABELS,DIR_VAL_IMAGES,DIR_VAL_LABELS]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# =========================================================
# CREATE DATASET YAML
# =========================================================

yaml_content = f"""
path: .

train: train/images
val: val/images

names:
  {CLASS_ID}: {CLASS_NAME}
"""

with open(f"{BASE_DIR}/dataset.yaml", "w") as f:
    f.write(yaml_content.strip())

# =========================================================
# LIST FILES
# =========================================================

VALID_BG = (".jpg", ".jpeg", ".png")
VALID_OBJ = (".png",)

backgrounds = [
    x for x in os.listdir(DIR_AMOSTRAS)
    if x.lower().endswith(VALID_BG)
]

objetos = [
    x for x in os.listdir(DIR_OBJETOS)
    if x.lower().endswith(VALID_OBJ)
]

if len(backgrounds) == 0:
    raise Exception("Nenhum background encontrado.")

if len(objetos) == 0:
    raise Exception("Nenhum objeto PNG encontrado.")

# =========================================================
# UTILS
# =========================================================

def get_bbox_alpha(img_rgba):
    alpha = np.array(img_rgba)[:, :, 3]
    ys, xs = np.where(alpha > 10)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = xs.min()
    y1 = ys.min()

    x2 = xs.max()
    y2 = ys.max()

    return x1, y1, x2, y2

# =========================================================
# GENERATE
# =========================================================

for i in range(TOTAL):

    # =====================================================
    # SELECT TRAIN / VAL
    # =====================================================

    usar_train = random.random() < TRAIN_SPLIT

    if usar_train:
        DIR_IMAGES = DIR_TRAIN_IMAGES
        DIR_LABELS = DIR_TRAIN_LABELS
    else:
        DIR_IMAGES = DIR_VAL_IMAGES
        DIR_LABELS = DIR_VAL_LABELS

    # =====================================================
    # LOAD BACKGROUND
    # =====================================================

    fundo_path = os.path.join(DIR_AMOSTRAS,random.choice(backgrounds))

    try:
        fundo = Image.open(fundo_path).convert("RGB")
    except Exception:
        print(f"[ERRO] Background inválido: {fundo_path}")
        continue
    bg_w, bg_h = fundo.size

    # =====================================================
    # NEGATIVE IMAGE
    # =====================================================

    if random.random() < NEGATIVE_RATIO:
        nome_img = f"neg_{i:06d}.jpg"
        nome_label = f"neg_{i:06d}.txt"
        caminho_img = os.path.join(DIR_IMAGES,nome_img)
        caminho_label = os.path.join(DIR_LABELS,nome_label)
        fundo.save(caminho_img,quality=95)
        # label vazio
        with open(caminho_label, "w") as f:
            pass

        print(f"[{i+1}/{TOTAL}] NEGATIVA")

        continue

    # =====================================================
    # OBJECTS COUNT
    # =====================================================

    num_objects = random.randint(1, 4)

    labels = []

    # =====================================================
    # ADD OBJECTS
    # =====================================================

    for _ in range(num_objects):
        objeto_path = os.path.join(DIR_OBJETOS,random.choice(objetos))

        try:
            objeto = Image.open(objeto_path).convert("RGBA")
        except Exception:
            print(f"[ERRO] PNG inválido: {objeto_path}")
            continue

        obj_w, obj_h = objeto.size

        # =================================================
        # SCALE
        # =================================================

        escala_max_w = bg_w / obj_w
        escala_max_h = bg_h / obj_h

        escala_max = min(escala_max_w,escala_max_h,0.6)

        escala_min = 0.08

        if escala_max <= escala_min:
            continue

        escala = random.uniform(escala_min,escala_max)

        novo_w = int(obj_w * escala)
        novo_h = int(obj_h * escala)

        if novo_w < 10 or novo_h < 10:
            continue

        objeto = objeto.resize((novo_w, novo_h),Image.Resampling.LANCZOS)

        # =================================================
        # ROTATION
        # =================================================

        angulo = random.randint(-40, 40)
        objeto = objeto.rotate(angulo,expand=True,resample=Image.Resampling.BICUBIC)

        # =================================================
        # BRIGHTNESS
        # =================================================

        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(objeto)
            objeto = enhancer.enhance(random.uniform(0.7, 1.4))

        # =================================================
        # CONTRAST
        # =================================================

        if random.random() < 0.4:
            enhancer = ImageEnhance.Contrast(objeto)
            objeto = enhancer.enhance(random.uniform(0.7, 1.4))

        # =================================================
        # BLUR
        # =================================================

        if random.random() < 0.3:
            objeto = objeto.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))

        # =================================================
        # BBOX REAL
        # =================================================

        bbox = get_bbox_alpha(objeto)

        if bbox is None:
            continue

        bx1, by1, bx2, by2 = bbox

        bbox_w = bx2 - bx1
        bbox_h = by2 - by1

        if bbox_w <= 5 or bbox_h <= 5:
            continue

        # =================================================
        # POSITION
        # =================================================

        obj_final_w, obj_final_h = objeto.size

        max_x = bg_w - obj_final_w
        max_y = bg_h - obj_final_h

        if max_x <= 0 or max_y <= 0:
            continue

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # =================================================
        # PASTE
        # =================================================

        fundo.paste(objeto,(x, y),objeto)

        # =================================================
        # ABSOLUTE BBOX
        # =================================================

        x1 = x + bx1
        y1 = y + by1

        x2 = x + bx2
        y2 = y + by2

        # clamp
        x1 = max(0, min(x1, bg_w - 1))
        y1 = max(0, min(y1, bg_h - 1))

        x2 = max(0, min(x2, bg_w - 1))
        y2 = max(0, min(y2, bg_h - 1))

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            continue

        # =================================================
        # YOLO FORMAT
        # =================================================

        x_center = ((x1 + x2) / 2) / bg_w
        y_center = ((y1 + y2) / 2) / bg_h

        width = bw / bg_w
        height = bh / bg_h

        # clamp normalize
        x_center = max(0, min(x_center, 1))
        y_center = max(0, min(y_center, 1))

        width = max(0, min(width, 1))
        height = max(0, min(height, 1))

        label = (f"{CLASS_ID} "f"{x_center:.6f} "f"{y_center:.6f} "f"{width:.6f} "f"{height:.6f}")

        labels.append(label)

    # =====================================================
    # SAVE IMAGE
    # =====================================================

    nome_img = f"img_{i:06d}.jpg"
    nome_label = f"img_{i:06d}.txt"

    caminho_img = os.path.join(
        DIR_IMAGES,
        nome_img
    )

    caminho_label = os.path.join(
        DIR_LABELS,
        nome_label
    )

    fundo.save(
        caminho_img,
        quality=95
    )

    # =====================================================
    # SAVE LABELS
    # =====================================================

    with open(caminho_label, "w") as f:
        f.write("\n".join(labels))

    print(
        f"[{i+1}/{TOTAL}] "
        f"OBJETOS={len(labels)}"
    )

print("\nDataset YOLO/MMYOLO gerado com sucesso.")