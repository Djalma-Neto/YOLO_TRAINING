# YOLOX migration layer

This project uses the official [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) detector, vendored under `third_party/YOLOX/`, with a local data-generation and training integration. New datasets go to `datasets/yolox_custom` by default. The current source tree is licensed under Apache-2.0; third-party files retain their upstream licenses. Existing commits in the Git history may contain an earlier MMYOLO/GPL version, but that code is removed from the current tracked tree.

## License

## License decision

The YOLOX source is Apache-2.0. The root `LICENSE` applies Apache-2.0 to the project-owned integration and scripts; `third_party/YOLOX/LICENSE` and its source headers preserve upstream notices. This permits commercial use of code under Apache-2.0, subject to its terms and notices. It does not establish rights for particular images, datasets, checkpoints, or separately installed dependencies. Review their licenses independently. The model zoo's code license does not by itself prove that a specific pretrained checkpoint is licensed for commercial use. Use random initialization or obtain and retain the checkpoint license. See the official [YOLOX license](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE), [custom-data guide](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/train_custom_data.md), and [ONNX Runtime guide](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/README.md).

## Install

Use the project's activated virtual environment. It must have a suitable PyTorch/torchvision build installed; CUDA training requires a compatible NVIDIA driver and CUDA-enabled PyTorch. The official YOLOX training implementation uses CUDA in core trainer paths, so this integration treats CUDA as a training requirement. Dataset generation, validation, conversion and ONNX inference can run on CPU.

```powershell
python -m pip install -r yolox_migration\requirements.txt
python -m pip install -e third_party\YOLOX
```

The vendored runtime subset is from the official YOLOX repository at revision `6ddff4824372906469a7fae2dc3206c7aa4bbaee`; see `third_party/YOLOX/UPSTREAM.md` for its provenance and included files.

The pip requirements for this adapter are permissively licensed in their upstream projects (Pillow HPND, NumPy BSD, PyYAML MIT, ONNX MIT, ONNX Runtime MIT). YOLOX's own runtime/training extras also include dependencies such as OpenCV, PyTorch, pycocotools, loguru and others; inspect their exact installed versions and license texts before redistributing an environment. Do not infer permission for a checkpoint from the code license.

## Data layout and generation

Edit `yolox_migration/config.yaml` to set paths, class names, image count, split fraction, seed, scale range, object-count range, maximum rotation, random/center placement, overlap allowance, architecture and training defaults. One-class mode accepts the existing flat `alvo/*.png` directory. For multiple classes, use class-named directories such as `alvo/prato/*.png` and `alvo/copo/*.png`, and list `classes: [prato, copo]` in the same order. Images in `amostras` are assigned to exactly one synthetic split; derivatives from a background won't cross train/validation.

```powershell
python -m yolox_migration.cli generate
python -m yolox_migration.cli validate
python -m yolox_migration.cli coco
```

Generation refuses a nonempty output by default. Use `--overwrite` only for a disposable migration output you intend to replace. It writes YOLO text labels, `dataset.yaml`, `classes.txt`, and a `split_manifest.json`. Alpha-threshold boxes are half-open, include the rotated visible-alpha extent, and are tightened after later pasted objects occlude prior objects. This remains bounding-box supervision: partially occluded objects are assigned the rectangle around their visible alpha pixels.

The generator only creates synthetic train/validation. For independent, manually annotated real imagery, import it into the held-out validation or test partition. Keep cameras/scenes and capture sessions grouped by partition to prevent real-image leakage:

```powershell
python -m yolox_migration.cli import-real --images C:\data\real_test\images --labels C:\data\real_test\labels --split test
python -m yolox_migration.cli validate
python -m yolox_migration.cli coco
```

Image and label files are copied; the input folders are not modified. Use an independently collected real `val` set for model selection and a separate real `test` set once, for final reporting. The repository currently has no known real labeled evaluation set, so the generated synthetic set cannot establish real-world restaurant performance.

## Train and resume

The vendored runtime is selected by default; `--yolox-root` is only needed to override it. The adapter writes a custom YOLOX experiment file in the output dataset directory with class count, model scale and resolution configured. COCO evaluation runs on validation each epoch and records standard detection metrics (including AP/mAP) and checkpoints in `third_party/YOLOX/YOLOX_outputs/`.

```powershell
python -m yolox_migration.cli train --model yolox-s --epochs 100 --batch-size 8 --size 640 --device cuda --pretrained C:\weights\yolox_s.pth
# Resume from an official YOLOX checkpoint:
python -m yolox_migration.cli train --resume C:\runs\custom_yolox-s\latest_ckpt.pth
```

`--device cpu` is rejected for training because the official trainer assumes CUDA; choose `cuda` or `auto` on a host with a working GPU. Transfer learning requires a compatible official YOLOX state dict. The final detection head differs when changing class count, so unmatched classifier/head keys may be skipped by YOLOX's checkpoint loader. Validate the load log and checkpoint provenance. Training is never started by generation or setup commands.

## Export and ONNX inference

Export uses the same generated experiment file as training, so the exported head class count and input resolution match the trained model. It enables the official `--decode_in_inference` option to produce the standard decoded output tensor `[1, N, 5 + classes]` expected by the inference adapter. That adapter applies objectness × class score, per-class NMS, maps coordinates back to the original image, and emits JSON with pixel `xyxy` boxes.

```powershell
python -m yolox_migration.cli export C:\runs\custom_yolox-s\best_ckpt.pth --model yolox-s --size 640
python -m yolox_migration.cli infer C:\runs\custom_yolox-s\best_ckpt.onnx C:\data\unseen.jpg --threshold 0.25 --json C:\data\predictions.json
```

If exporting with a modified/custom YOLOX head or output decoding, verify and adapt `onnx_inference.py`; it intentionally rejects output tensors whose class dimension differs from the configured classes. Keep the test set outside training, report precision/recall and COCO mAP from real labeled data, and compare against prior models on the same untouched examples. Architectural changes do not imply score equivalence.

## Tests

Run the focused tests without downloads or training:

```powershell
python -m unittest discover -s yolox_migration\tests -v
```

No full training or model download is required for these tests.
