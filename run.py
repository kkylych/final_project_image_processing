"""
Entry point for the handwriting recognition project.

Usage:
  python run.py train              # download dataset and train the model
  python run.py infer              # run inference on the validation set (requires trained model)
  python run.py infer path/to/img  # run inference on a single image file
"""

import os
import sys
import argparse

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "App", "handwriting_recognition")


def run_train():
    """Launch training from the App directory so all relative paths resolve correctly."""
    train_script = os.path.join(APP_DIR, "train.py")
    print(f"Starting training: {train_script}")
    # Use subprocess so the script runs in its own interpreter context
    import subprocess
    result = subprocess.run([sys.executable, train_script], check=False)
    return result.returncode


def _ensure_dataset():
    """Download and extract the IAM Words dataset if it is not present."""
    import tarfile
    import zipfile
    from urllib.request import urlopen, Request
    from io import BytesIO
    from tqdm import tqdm

    dataset_path = os.path.join(APP_DIR, "Datasets", "IAM_Words")
    words_path   = os.path.join(dataset_path, "words")

    if os.path.exists(words_path) and os.listdir(words_path):
        return  # already downloaded

    os.makedirs(dataset_path, exist_ok=True)
    print("Dataset not found locally — downloading IAM Words (~60 MB)...")

    req = Request("https://git.io/J0fjL", headers={"User-Agent": "Mozilla/5.0"})
    response = urlopen(req)
    total = int(response.headers.get("Content-Length", 0))
    data  = b""
    bar   = tqdm(total=total, unit="B", unit_scale=True, desc="Downloading")
    while True:
        block = response.read(1024 * 1024)
        if not block:
            break
        data += block
        bar.update(len(block))
    bar.close()

    if data[:2] == b"PK":
        with zipfile.ZipFile(BytesIO(data)) as zf:
            zf.extractall(os.path.join(APP_DIR, "Datasets"))
    elif data[:2] in (b"\x1f\x8b", b"BZ"):
        with tarfile.open(fileobj=BytesIO(data)) as tf:
            tf.extractall(dataset_path)
    else:
        print(f"ERROR: Unrecognised archive format (first bytes: {data[:8].hex()})")
        return

    tgz = os.path.join(dataset_path, "words.tgz")
    if os.path.exists(tgz):
        print("Extracting words.tgz ...")
        os.makedirs(words_path, exist_ok=True)
        with tarfile.open(tgz) as tf:
            tf.extractall(words_path)

    print("Dataset ready.\n")


def run_infer(image_path=None):
    """Run inference. If image_path is given, predict that single image."""
    import cv2
    import numpy as np
    from mltu.configs import BaseModelConfigs
    from mltu.inferenceModel import OnnxInferenceModel
    from mltu.utils.text_utils import ctc_decoder

    models_base = os.path.join(APP_DIR, "Models", "handwriting_recognition")
    if not os.path.exists(models_base):
        print(f"ERROR: No trained models found at:\n  {models_base}")
        print("Run training first:  python run.py train")
        return 1

    # Pick the most recently trained model
    candidates = sorted(
        [d for d in os.listdir(models_base) if os.path.isdir(os.path.join(models_base, d))],
        reverse=True,
    )
    if not candidates:
        print(f"ERROR: No model subdirectory found in {models_base}")
        return 1

    model_dir = os.path.join(models_base, candidates[0])
    configs_path = os.path.join(model_dir, "configs.yaml")
    onnx_path = os.path.join(model_dir, "model.onnx")

    if not os.path.exists(configs_path):
        print(f"ERROR: configs.yaml not found in {model_dir}")
        return 1
    if not os.path.exists(onnx_path):
        print(f"ERROR: model.onnx not found in {model_dir}")
        print("Training may have completed but ONNX export failed. Check training logs.")
        return 1

    configs = BaseModelConfigs.load(configs_path)

    class ImageToWordModel(OnnxInferenceModel):
        def __init__(self, char_list, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.char_list = char_list

        def predict(self, image: np.ndarray):
            image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
            image_pred = np.expand_dims(image, axis=0).astype(np.float32)
            preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
            return ctc_decoder(preds, self.char_list)[0]

    model = ImageToWordModel(model_path=onnx_path, char_list=configs.vocab)

    if image_path:
        # Single image inference
        if not os.path.exists(image_path):
            print(f"ERROR: Image not found: {image_path}")
            return 1
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not read image: {image_path}")
            return 1
        prediction = model.predict(img)
        print(f"Image : {image_path}")
        print(f"Result: {prediction}")
        return 0

    # Batch inference on validation CSV
    _ensure_dataset()
    import pandas as pd
    from tqdm import tqdm

    val_csv = os.path.join(model_dir, "val.csv")
    if not os.path.exists(val_csv):
        print(f"ERROR: val.csv not found at {val_csv}")
        return 1

    from mltu.utils.text_utils import get_cer
    df = pd.read_csv(val_csv).values.tolist()
    cer_list = []
    for row in tqdm(df, desc="Inference"):
        image_file, label = row[0], row[1]
        if not os.path.isabs(image_file):
            image_file = os.path.join(APP_DIR, image_file)
        img = cv2.imread(image_file.replace("\\", "/"))
        if img is None:
            continue
        pred = model.predict(img)
        if not isinstance(pred, str):
            pred = ""
        cer = get_cer(pred, label)
        cer_list.append(cer)
        print(f"  label={label:20s}  pred={pred:20s}  CER={cer:.3f}")

    if cer_list:
        print(f"\nAverage CER: {sum(cer_list)/len(cer_list):.4f}  over {len(cer_list)} samples")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Handwriting Recognition")
    parser.add_argument("mode", choices=["train", "infer"], help="train or infer")
    parser.add_argument("image", nargs="?", default=None, help="Path to image (infer mode only)")
    args = parser.parse_args()

    if args.mode == "train":
        sys.exit(run_train())
    elif args.mode == "infer":
        sys.exit(run_infer(args.image))


if __name__ == "__main__":
    main()
