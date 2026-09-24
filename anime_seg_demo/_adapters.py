"""Private adapters using verified inference settings."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from ._settings import get_settings


def _state_dict(path):
    if path.suffix != ".safetensors":
        raise ValueError("Only .safetensors checkpoints are supported")
    state = load_file(str(path), device="cpu")
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("A complete tensor state_dict is required")
    return state


class _Adapter:
    def __init__(self, model_type, checkpoint):
        self.model_type = model_type
        self.device = torch.device("cpu")
        self.settings = get_settings(model_type)
        self.config = self.settings["config"]
        self.num_classes = self.config["num_labels"]
        self.preprocess_config = self.settings["preprocessing"]
        state = _state_dict(checkpoint)

        if model_type == "dinov2_unetpp":
            from ._dinov2 import DINOv2ForSegmentation
            with torch.device("meta"):
                self.model = DINOv2ForSegmentation(self.config)
        elif model_type == "afs":
            from ._afs import AFSModel
            self.model = AFSModel(num_classes=self.num_classes)
        elif model_type == "thinhrnet_ocr":
            from ._thinhrnet.model import ThinHRNetOCR
            self.model = ThinHRNetOCR(**self.config["model_kwargs"], pretrained=False)
        elif model_type == "oneformer":
            from transformers import OneFormerConfig, OneFormerForUniversalSegmentation
            with torch.device("meta"):
                self.model = OneFormerForUniversalSegmentation(
                    OneFormerConfig.from_dict(self.config)
                )
        elif model_type == "mask2former":
            from ._mask2former import Mask2FormerAnimeSegModel
            if self.num_classes != 12:
                raise ValueError("This Mask2Former demo requires the 12-class checkpoint")
            with torch.device("meta"):
                self.model = Mask2FormerAnimeSegModel(self.config)
        else:
            raise ValueError(f"Unsupported model: {model_type}")

        target = self.model.model if model_type == "mask2former" else self.model
        target.load_state_dict(state, strict=True, assign=True)
        self.model.eval()
        if any(p.is_meta for p in self.model.parameters()):
            raise RuntimeError("Checkpoint did not materialize every model parameter")

    def _preprocess(self, image):
        cfg = self.preprocess_config
        height, width = cfg["size"]
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        if cfg["image_mean"] is not None:
            pixels = (pixels - np.asarray(cfg["image_mean"], dtype=np.float32)) / np.asarray(
                cfg["image_std"], dtype=np.float32
            )
        return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).unsqueeze(0).to(self.device)

    def predict(self, image):
        pixels = self._preprocess(image)
        if self.model_type == "oneformer":
            task = torch.tensor(
                [self.settings["task_input_ids"]], device=self.device, dtype=torch.long
            )
            outputs = self.model(pixel_values=pixels, task_inputs=task)
            # OneFormerImageProcessor semantic postprocessing from the training path.
            classes = outputs.class_queries_logits.softmax(-1)[..., :-1]
            masks = outputs.masks_queries_logits.sigmoid()
            logits = torch.einsum("bqc,bqhw->bchw", classes, masks)
            logits = F.interpolate(
                logits, size=pixels.shape[-2:], mode="bilinear", align_corners=False
            )
        elif self.model_type == "mask2former":
            # The unversioned AnimeSeg model resizes query logits before sigmoid.
            logits = self.model(pixels)["semantic_logits"]
        elif self.model_type == "thinhrnet_ocr":
            logits = self.model(pixels)["logits"]
            logits = F.interpolate(
                logits, size=pixels.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            logits = self.model(pixels)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{self.model_type} produced non-finite semantic scores")
        return logits.argmax(dim=1)[0]


def load_adapter(model_type, checkpoint):
    return _Adapter(model_type, checkpoint)
