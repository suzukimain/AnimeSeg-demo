"""One image in, an H x W integer class-ID mask out."""
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

DEFAULT_REPO_ID = "suzukimain/AnimeSeg-demo"


def _checkpoint_from_catalog(path, model_type):
    from ._settings import get_settings

    catalog = json.loads(path.read_text(encoding="utf-8"))
    entries = [entry for entry in catalog["models"]
               if entry["FilePath"].split("/")[0] == model_type]
    if len(entries) != 1:
        raise ValueError(f"Expected one catalog entry for {model_type}")
    entry = entries[0]
    filename = entry["FilePath"]
    relative = PurePosixPath(filename)
    if (relative.is_absolute() or len(relative.parts) != 2
            or ".." in relative.parts or "\\" in filename
            or relative.suffix != ".safetensors"):
        raise ValueError("Checkpoint must be a safetensors file inside its model folder")
    expected_arch = "dinov2" if model_type == "dinov2_unetpp" else model_type
    size = get_settings(model_type)["preprocessing"]["size"]
    if entry["Architecture"] != expected_arch or size != [entry["TrainImageSize"]] * 2:
        raise ValueError(f"Catalog settings do not match {model_type}")
    return filename


class AnimeSegPipelineDemo:
    def __init__(self, adapter):
        self._adapter = adapter
        self.eval()

    @property
    def model_type(self):
        return self._adapter.model_type

    @property
    def num_classes(self):
        return self._adapter.num_classes

    @property
    def device(self):
        return self._adapter.device

    def eval(self):
        self._adapter.model.eval()
        return self

    def to(self, device):
        self._adapter.model.to(device)
        self._adapter.device = torch.device(device)
        return self.eval()

    def __call__(self, image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                source = source.convert("RGB")
        elif isinstance(image, Image.Image):
            source = image.convert("RGB")
        else:
            raise TypeError("Input must be an image path or PIL.Image.Image")

        self.eval()
        with torch.inference_mode():
            prediction = self._adapter.predict(source)
        mask = prediction.detach().cpu().numpy().astype(np.int32)
        if mask.ndim != 2 or mask.min() < 0 or mask.max() >= self.num_classes:
            raise RuntimeError("Model returned an invalid class-ID mask")
        if mask.shape != (source.height, source.width):
            mask = np.asarray(
                Image.fromarray(mask).resize(source.size, Image.Resampling.NEAREST),
                dtype=np.int32,
            )
        return mask

    @classmethod
    def _from_model(
        cls, model_type, model_dir=None, *, repo_id=DEFAULT_REPO_ID,
        revision=None, local_files_only=False, cache_dir=None,
    ):
        from ._adapters import load_adapter

        if model_dir is None:
            from huggingface_hub import hf_hub_download

            options = dict(repo_id=repo_id, revision=revision, cache_dir=cache_dir,
                           local_files_only=local_files_only)
            catalog_path = Path(hf_hub_download(filename="config.json", **options))
            filename = _checkpoint_from_catalog(catalog_path, model_type)
            checkpoint = Path(hf_hub_download(filename=filename, **options))
        else:
            root = Path(model_dir)
            if not (root / "config.json").is_file():
                root = root.parent
            filename = _checkpoint_from_catalog(root / "config.json", model_type)
            checkpoint = root / filename
        return cls(load_adapter(model_type, checkpoint))

    @classmethod
    def from_dinov2_unetpp(cls, model_dir=None, **kwargs):
        return cls._from_model("dinov2_unetpp", model_dir, **kwargs)

    @classmethod
    def from_oneformer(cls, model_dir=None, **kwargs):
        return cls._from_model("oneformer", model_dir, **kwargs)

    @classmethod
    def from_thinhrnet_ocr(cls, model_dir=None, **kwargs):
        return cls._from_model("thinhrnet_ocr", model_dir, **kwargs)

    @classmethod
    def from_afs(cls, model_dir=None, **kwargs):
        return cls._from_model("afs", model_dir, **kwargs)

    @classmethod
    def from_mask2former(cls, model_dir=None, **kwargs):
        return cls._from_model("mask2former", model_dir, **kwargs)
