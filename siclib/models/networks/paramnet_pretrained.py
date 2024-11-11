"""Interface for PerspectiveFields inference."""

from siclib.models.base_model import BaseModel
from siclib.models.paramnet_extractor import ParamNet


# mypy: ignore-errors
class ParamNetPretrained(BaseModel):
    """ParamNet pretrained model."""

    default_conf = {
        "model_weights": "openpano",
    }

    def _init(self, conf):
        """Initialize pretrained ParamNet model."""
        self.model = ParamNet(weights=conf.model_weights)

    def _forward(self, data):
        """Forward pass."""
        return self.model.calibrate(data["image"])

    def metrics(self, pred, data):
        """Compute metrics."""
        raise NotImplementedError("ParamNetPretrained does not support metrics computation.")

    def loss(self, pred, data):
        """Compute loss."""
        raise NotImplementedError("ParamNetPretrained does not support loss computation.")
