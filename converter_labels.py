import os
import json
from PIL import Image

def yolo_to_coco(images_dir, labels_dir, output_json, class_names):
    coco_format = {"images": [], "annotations": [], "categories": []}
    
    for i, name in enumerate(class_names):
        coco_format["categories"].append({"id": i, "name": name})
        
    ann_id = 0
    img_id = 0
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(images_dir, img_name)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except:
            continue
            
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:])
                        
                        # Converter YOLO (centro e tamanho normalizado) para COCO (topo esquerdo e pixels)
                        w_px, h_px = w * width, h * height
                        x_min = (x_c * width) - (w_px / 2)
                        y_min = (y_c * height) - (h_px / 2)
                        
                        coco_format["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": c_id,
                            "bbox": [x_min, y_min, w_px, h_px],
                            "area": w_px * h_px,
                            "iscrowd": 0
                        })
                        ann_id += 1
        img_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

# Gerar para treino e validação (Assumindo que a pasta datasets está na raiz)
yolo_to_coco("./datasets/custom/train/images", "./datasets/custom/train/labels", "./datasets/custom/train.json", ["objeto"])
yolo_to_coco("./datasets/custom/val/images", "./datasets/custom/val/labels", "./datasets/custom/val.json", ["objeto"])
print("Ficheiros JSON criados com sucesso!")