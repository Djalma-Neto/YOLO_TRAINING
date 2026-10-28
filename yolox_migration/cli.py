from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

from .pipeline import generate, import_real, validate_dataset, write_coco

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def config(path):
    try:
        import yaml
    except ImportError:
        cfg = {}
        active_list = None
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].rstrip()
            if not line.strip(): continue
            stripped = line.strip()
            if stripped.startswith("- ") and active_list:
                cfg[active_list].append(_yaml_scalar(stripped[2:].strip()))
                continue
            if line[:1].isspace(): raise ValueError(f"Unsupported YAML structure: {line}")
            if ":" not in line: raise ValueError(f"Invalid YAML setting: {line}")
            key, value = line.split(":", 1); value = value.strip()
            if value:
                cfg[key.strip()] = _yaml_scalar(value); active_list = None
            else:
                cfg[key.strip()] = []; active_list = key.strip()
    else:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict): raise ValueError("Config must be a YAML mapping")
    for key in ("backgrounds", "objects", "output"):
        cfg[key] = (ROOT / cfg[key]).resolve() if not Path(cfg[key]).is_absolute() else Path(cfg[key])
    return cfg


def _yaml_scalar(value):
    if value.lower() in {"true", "false"}: return value.lower() == "true"
    try: return ast.literal_eval(value)
    except (ValueError, SyntaxError): return value


def exp_source(cfg, model, epochs, size):
    model_dims = {"yolox-nano": (.33, .25), "yolox-tiny": (.33, .375), "yolox-s": (.33, .50),
                  "yolox-m": (.67, .75), "yolox-l": (1.0, 1.0), "yolox-x": (1.33, 1.25)}
    if model not in model_dims: raise ValueError(f"Unsupported architecture {model}")
    depth, width = model_dims[model]
    ds = repr(str(cfg["output"]))
    return f'''from yolox.exp import Exp as BaseExp
from yolox.data import COCODataset, TrainTransform, ValTransform
class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = {len(cfg['classes'])}
        self.depth, self.width = {depth}, {width}
        self.input_size = ({size}, {size})
        self.test_size = ({size}, {size})
        self.max_epoch = {epochs}
        self.data_num_workers = 2
        self.data_dir = {ds}
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.exp_name = "custom_{model}"
        self.enable_mixup = False
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.eval_interval = 1
    def get_dataset(self, cache=False, cache_type="ram"):
        return COCODataset(data_dir=self.data_dir, json_file=self.train_ann,
            name="train/images", img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache, cache_type=cache_type)
    def get_eval_dataset(self, **kwargs):
        return COCODataset(data_dir=self.data_dir, json_file=self.val_ann, name="val/images",
            img_size=self.test_size, preproc=ValTransform())
'''


def main():
    p = argparse.ArgumentParser(description="Synthetic-data and official YOLOX workflow")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate"); gen.add_argument("--overwrite", action="store_true")
    sub.add_parser("validate")
    sub.add_parser("coco")
    real = sub.add_parser("import-real", help="Copy manually labeled real images to held-out val/test")
    real.add_argument("--images", type=Path, required=True); real.add_argument("--labels", type=Path, required=True)
    real.add_argument("--split", choices=["val", "test"], required=True)
    train = sub.add_parser("train")
    train.add_argument("--model", choices=["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"])
    train.add_argument("--epochs", type=int); train.add_argument("--batch-size", type=int)
    train.add_argument("--size", type=int); train.add_argument("--device", choices=["auto", "cuda", "cpu"])
    train.add_argument("--pretrained", type=Path); train.add_argument("--resume", type=Path)
    train.add_argument("--yolox-root", type=Path, help="Override the vendored YOLOX runtime path")
    export = sub.add_parser("export")
    export.add_argument("checkpoint", type=Path); export.add_argument("--model", choices=["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"])
    export.add_argument("--yolox-root", type=Path, help="Override the vendored YOLOX runtime path"); export.add_argument("--size", type=int)
    infer = sub.add_parser("infer")
    infer.add_argument("model", type=Path); infer.add_argument("image", type=Path)
    infer.add_argument("--threshold", type=float, default=.25); infer.add_argument("--json", type=Path)
    args = p.parse_args(); cfg = config(args.config)
    if args.cmd == "generate":
        generate(cfg["backgrounds"], cfg["objects"], cfg["output"], classes=cfg["classes"],
            count=cfg["images"], train_fraction=cfg["train_fraction"],
            negative_fraction=cfg["negative_fraction"], seed=cfg["seed"], overwrite=args.overwrite,
            min_scale=cfg.get("min_scale", .08), max_scale=cfg.get("max_scale", .6),
            min_objects=cfg.get("min_objects", 1), max_objects=cfg.get("max_objects", 4),
            max_rotation_degrees=cfg.get("max_rotation_degrees", 40),
            placement=cfg.get("placement", "random"), allow_overlap=cfg.get("allow_overlap", True))
        print(f"Generated dataset at {cfg['output']}")
    elif args.cmd == "validate":
        errors = validate_dataset(cfg["output"], cfg["classes"])
        if errors: print("\n".join(errors)); raise SystemExit(1)
        print("Dataset labels and partition structure are valid")
    elif args.cmd == "coco":
        errors = validate_dataset(cfg["output"], cfg["classes"])
        if errors: print("\n".join(errors)); raise SystemExit(1)
        write_coco(cfg["output"], cfg["classes"])
        print(f"COCO annotations written to {cfg['output'] / 'annotations'}")
    elif args.cmd == "import-real":
        import_real(args.images.resolve(), args.labels.resolve(), cfg["output"], args.split)
        print(f"Copied real {args.split} images into {cfg['output'] / args.split}")
    elif args.cmd in ("train", "export"):
        yroot = (args.yolox_root or ROOT / "third_party" / "YOLOX").resolve()
        if not (yroot / "tools" / ("train.py" if args.cmd == "train" else "export_onnx.py")).is_file():
            raise SystemExit("Vendored YOLOX runtime is incomplete; restore third_party/YOLOX or pass --yolox-root")
        if args.cmd == "train":
            if args.device == "cpu": raise SystemExit("Official YOLOX trainer requires CUDA; CPU is supported for ONNX inference.")
            try:
                import torch
            except ImportError as exc:
                raise SystemExit("Install CUDA-enabled PyTorch in the YOLOX environment before training") from exc
            if not torch.cuda.is_available():
                raise SystemExit("YOLOX training requires an available CUDA GPU; CPU inference remains supported")
            errors = validate_dataset(cfg["output"], cfg["classes"])
            if errors: print("\n".join(errors)); raise SystemExit(1)
            write_coco(cfg["output"], cfg["classes"])
            model = args.model or cfg["model"]; epochs = args.epochs or cfg["epochs"]
            size = args.size or cfg["image_size"]; batch = args.batch_size or cfg["batch_size"]
            exp = cfg["output"] / f"exp_{model}.py"
            # The generated exp is contained within the dataset output and can be reproduced.
            exp.write_text(exp_source(cfg, model, epochs, size), encoding="utf-8")
            cmd = [sys.executable, str(yroot / "tools" / "train.py"), "-f", str(exp), "-d", "1", "-b", str(batch)]
            checkpoint = args.resume or args.pretrained or (Path(cfg["pretrained"]) if cfg.get("pretrained") else None)
            if args.resume: cmd.append("--resume")
            if checkpoint: cmd += ["-c", str(checkpoint.resolve())]
            subprocess.run(cmd, cwd=yroot, check=True)
        else:
            model = args.model or cfg["model"]
            size = args.size or cfg["image_size"]
            cfg["output"].mkdir(parents=True, exist_ok=True)
            exp = cfg["output"] / f"exp_{model}.py"
            exp.write_text(exp_source(cfg, model, cfg["epochs"], size), encoding="utf-8")
            cmd = [sys.executable, str(yroot / "tools" / "export_onnx.py"), "-f", str(exp),
                   "-c", str(args.checkpoint.resolve()), "--output-name", str(args.checkpoint.with_suffix(".onnx").resolve()),
                   "--decode_in_inference"]
            subprocess.run(cmd, cwd=yroot, check=True)
    elif args.cmd == "infer":
        from .onnx_inference import predict
        predictions = predict(args.model, args.image, cfg["classes"], args.threshold)
        import json
        payload = [{"class_id": x[0], "class": cfg["classes"][x[0]], "score": x[1], "bbox_xyxy": x[2]} for x in predictions]
        print(json.dumps(payload, indent=2))
        if args.json: args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
