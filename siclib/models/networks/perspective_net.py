import logging

from siclib.models import get_model
from siclib.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class PerspectiveNet(BaseModel):
    default_conf = {
        "backbone": {"name": "encoders.mix_vit"},
        "ll_enc": {"name": "encoders.low_level_encoder"},
        "perspective_decoder": {"name": "decoders.perspective_decoder"},
        "param_net": None,
        "param_opt": None,
        "weights": None,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        logger.debug(f"Initializing PerspectiveNet with {conf}")
        # self.backbone = get_model_from_conf(conf.backbone)
        self.backbone = get_model(conf.backbone["name"])(conf.backbone)
        self.ll_enc = get_model(conf.ll_enc["name"])(conf.ll_enc) if conf.ll_enc else None
        self.persformer_heads = get_model(conf.perspective_decoder["name"])(
            conf.perspective_decoder
        )
        self.param_net = (
            get_model(conf.param_net["name"])(conf.param_net) if conf.param_net else None
        )
        self.param_opt = (
            get_model(conf.param_opt["name"])(conf.param_opt) if conf.param_opt else None
        )

    def _forward(self, data):
        backbone_out = self.backbone(data)
        features = {"hl": backbone_out["features"], "padding": backbone_out.get("padding", None)}

        if self.ll_enc is not None:
            features["ll"] = self.ll_enc(data)["features"]  # low level features

        out = self.persformer_heads({"features": features})
        out |= {
            k: data[k]
            for k in ["image", "scales", "prior_gravity", "prior_focal", "prior_k1"]
            if k in data
        }

        if self.param_net is not None:
            out |= self.param_net(out)

        if self.param_opt is not None:
            out |= self.param_opt(out)

        return out

    def loss(self, pred, data):
        losses, metrics = self.persformer_heads.loss(pred, data)
        total = losses["perspective_total"]

        if self.param_net is not None:
            param_losses, param_metrics = self.param_net.loss(pred, data)
            losses |= param_losses
            metrics |= param_metrics
            total = total + param_losses["param_total"]

        if self.param_opt is not None:
            opt_losses, param_metrics = self.param_opt.loss(pred, data)
            losses |= opt_losses
            metrics |= param_metrics
            total = total + opt_losses["opt_param_total"]

        losses["total"] = total
        return losses, metrics
