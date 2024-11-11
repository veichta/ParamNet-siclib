"""Simple interface for ParamNet model."""

from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf
from torch.nn.functional import interpolate

from siclib.models import get_model
from siclib.utils.image import ImagePreprocessor, load_image


class ParamNet(torch.nn.Module):
    """Simple interface for ParamNet model."""

    base_conf = {
        "name": "networks.perspective_net",
        "backbone": {"name": "encoders.mix_vit"},
        "perspective_decoder": {
            "name": "decoders.perspective_decoder",
            "up_decoder": {
                "name": "decoders.up_decoder",
                "loss_type": "l2",
                "use_uncertainty_loss": False,
                "decoder": {
                    "name": "decoders.persformer_decoder",
                    "predict_uncertainty": False,
                    "use_original_architecture": True,
                },
            },
            "latitude_decoder": {
                "name": "decoders.latitude_decoder",
                "loss_type": "l2",
                "use_uncertainty_loss": False,
                "use_tanh": False,
                "decoder": {
                    "name": "decoders.persformer_decoder",
                    "predict_uncertainty": False,
                    "use_original_architecture": True,
                },
            },
        },
        "ll_enc": {"name": "encoders.low_level_encoder", "keep_resolution": False},
        "param_net": {"name": "networks.param_net", "original_weights": True},
    }

    def __init__(self, weights: str = "openpano", **conf):
        """Initialize the model with optional config overrides."""
        super().__init__()

        if weights == "openpano":
            url = "https://www.polybox.ethz.ch/index.php/s/N91Kws2SkSstxeJ/download"
        elif weights == "360cities":
            url = (
                "https://www.dropbox.com/scl/fi/9xmt4pdx50ida61jstyua/paramnet_360cities_edina_rpf"
                + ".pth?rlkey=av94fij0wk4sqkoot5y11sfc4&e=1&st=uger5gb8&dl=1"
            )
        else:
            raise ValueError(f"Unknown weights '{weights}', must be 'openpano' or '360cities'.")

        model_dir = f"{torch.hub.get_dir()}/paramnet/"
        state_dict = torch.hub.load_state_dict_from_url(
            url, model_dir, map_location="cpu", file_name=f"paramnet-{weights}.tar"
        )

        self.model_conf = OmegaConf.create(self.base_conf)
        self.conf = OmegaConf.create({**self.model_conf, **conf})

        self.preprocess_conf = {"resize": 320, "edge_divisible_by": 32}
        self.model = get_model(self.conf.name)(self.conf)
        self.model.flexible_load(state_dict["model"])
        self.model.eval()

        self.image_processor = ImagePreprocessor({**self.preprocess_conf})

    def load_image(self, path: Path) -> torch.Tensor:
        """Load image from path."""
        return load_image(path)

    @torch.no_grad()
    def calibrate(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform calibration with online resizing.

        Assumes input image is in range [0, 1] and in RGB format.

        Args:
            img (torch.Tensor): Input image, shape (C, H, W) or (1, C, H, W)
            priors (Dict[str, torch.Tensor], optional): Prior parameters. Defaults to {}.
            conf (Dict): Additional configuration for image preprocessing.

        Returns:
            Dict[str, torch.Tensor]: camera and gravity vectors and uncertainties.
        """
        if len(img.shape) == 3:
            img = img[None]  # add batch dim
        assert len(img.shape) == 4 and img.shape[0] == 1

        img_data = self.image_processor(img)
        out = self.model.forward(img_data)

        gravity, camera = out["gravity"], out["camera"]
        camera = camera.undo_scale_crop(img_data)

        w, h = camera.size.unbind(-1)
        h, w = h.int().item(), w.int().item()

        for k in ["latitude_field", "up_field"]:
            out[k] = interpolate(out[k], size=(h, w), mode="bilinear")

        return {
            "camera": camera,
            "gravity": gravity,
            "up_field": out["up_field"],
            "latitude_field": out["latitude_field"],
        }
