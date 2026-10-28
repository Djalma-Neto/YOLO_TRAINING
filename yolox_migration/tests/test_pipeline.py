import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from yolox_migration.cli import config
from yolox_migration.pipeline import alpha_bbox, generate, import_real, validate_dataset, write_coco
from yolox_migration.onnx_inference import postprocess, preprocess


class PipelineTests(unittest.TestCase):
    def test_default_config_loads(self):
        cfg = config(Path(__file__).parents[1] / "config.yaml")
        self.assertEqual(cfg["classes"], ["objeto"])
        self.assertEqual(cfg["images"], 1000)

    def test_alpha_bbox_half_open(self):
        image = Image.new("RGBA", (12, 9)); ImageDraw.Draw(image).rectangle((3, 2, 7, 5), fill=(255, 0, 0, 255))
        self.assertEqual(alpha_bbox(image), (3, 2, 8, 6))
        self.assertIsNone(alpha_bbox(Image.new("RGBA", (3, 3))))

    def test_generation_labels_splits_and_coco(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bg = root / "bg"; objs = root / "obj"; out = root / "out"
            bg.mkdir(); objs.mkdir()
            for i in range(5): Image.new("RGB", (64, 64), (i * 20, 30, 40)).save(bg / f"bg{i}.jpg")
            obj = Image.new("RGBA", (16, 16)); ImageDraw.Draw(obj).ellipse((2, 3, 13, 14), fill=(255, 0, 0, 255)); obj.save(objs / "thing.png")
            generate(bg, objs, out, classes=["thing"], count=18, train_fraction=.6,
                     negative_fraction=.2, seed=42)
            self.assertEqual(validate_dataset(out, ["thing"]), [])
            manifest = json.loads((out / "split_manifest.json").read_text())
            train = {k for k, v in manifest["background_split"].items() if v == "train"}
            val = {k for k, v in manifest["background_split"].items() if v == "val"}
            self.assertFalse(train & val)
            write_coco(out, ["thing"])
            payload = json.loads((out / "annotations" / "instances_train2017.json").read_text())
            self.assertTrue(payload["images"])
            self.assertEqual(payload["categories"][0]["id"], 1)
            no_overlap = root / "no_overlap"
            generate(bg, objs, no_overlap, classes=["thing"], count=1, train_fraction=.6,
                     negative_fraction=0, seed=1, min_objects=2, max_objects=2,
                     placement="center", allow_overlap=False)
            labels = next((no_overlap / "train" / "labels").glob("*.txt"))
            self.assertLessEqual(len(labels.read_text().splitlines()), 1)

    def test_real_import_does_not_overwrite_and_onnx_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); images = root / "images"; labels = root / "labels"; out = root / "data"
            images.mkdir(); labels.mkdir()
            Image.new("RGB", (100, 50)).save(images / "real.jpg"); (labels / "real.txt").write_text("0 0.5 0.5 0.2 0.4\n")
            import_real(images, labels, out, "test")
            with self.assertRaises(FileExistsError): import_real(images, labels, out, "test")
        source = Image.new("RGB", (100, 50))
        tensor, ratio = preprocess(source, (100, 100))
        self.assertEqual(tensor.shape, (1, 3, 100, 100)); self.assertEqual(ratio, 1.0)
        raw = np.array([[[50, 50, 20, 10, .9, .8]]], dtype=np.float32)
        result = postprocess(raw, 1.0, (source.height, source.width, 3), threshold=.2, num_classes=1)
        self.assertEqual(result[0][0], 0); self.assertAlmostEqual(result[0][1], .72, places=5)
        self.assertEqual(result[0][2], [40.0, 45.0, 60.0, 50.0])

    def test_onnx_session_load_and_image_inference_when_runtime_is_installed(self):
        try:
            import onnx
            from onnx import helper, TensorProto
            import onnxruntime
        except ImportError:
            self.skipTest("onnx and onnxruntime are optional in the lightweight test environment")
        from yolox_migration.onnx_inference import predict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model_path = root / "constant.onnx"; image_path = root / "image.jpg"
            Image.new("RGB", (64, 48), (80, 90, 100)).save(image_path)
            output = helper.make_tensor("pred", TensorProto.FLOAT, [1, 1, 6], [32, 24, 10, 8, .9, .8])
            graph = helper.make_graph([helper.make_node("Constant", [], ["detections"], value=output)],
                "constant_yolox", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
                [helper.make_tensor_value_info("detections", TensorProto.FLOAT, [1, 1, 6])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 9
            onnx.save(model, model_path)
            result = predict(model_path, image_path, ["thing"], threshold=.2)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0][1], .72, places=5)


if __name__ == "__main__": unittest.main()
