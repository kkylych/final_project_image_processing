import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    _HERE = os.path.dirname(os.path.abspath(__file__))
    models_base = os.path.join(_HERE, "Models", "handwriting_recognition")

    # Auto-detect most recently trained model directory
    if not os.path.exists(models_base):
        raise FileNotFoundError(
            f"No trained models found at {models_base}. Run train.py first."
        )
    candidates = sorted(
        [d for d in os.listdir(models_base) if os.path.isdir(os.path.join(models_base, d))],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No model subdirectories found in {models_base}. Run train.py first."
        )
    model_dir = os.path.join(models_base, candidates[0])

    configs = BaseModelConfigs.load(os.path.join(model_dir, "configs.yaml"))

    model = ImageToWordModel(model_path=model_dir, char_list=configs.vocab)

    df = pd.read_csv(os.path.join(model_dir, "val.csv")).values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accum_cer.append(cer)

        # resize by 4x
        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Average CER: {np.average(accum_cer)}")