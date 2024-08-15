import torch
from torch.nn import functional as F

from siclib.geometry.camera import camera_models
from siclib.geometry.gravity import Gravity
from siclib.models import get_model
from siclib.models.base_model import BaseModel
from siclib.models.utils.metrics import dist_error, pitch_error, roll_error, vfov_error

# flake8: noqa
# mypy: ignore-errors


class ParamNet(BaseModel):
    default_conf = {
        "backbone": {"name": "encoders.convnext"},
        "loss_weight": 1.0,
        "recover_pp": False,
        "original_weights": False,
        "estimate_distortions": False,
        "distortion_range": [-0.7, 0.7],
        "loss_scales": [1.0, 1.0, 1.0, 1.0],
    }

    required_data_keys = ["up_field", "latitude_field"]

    def _init(self, conf):
        self.backbone = get_model(self.conf.backbone.name)(self.conf.backbone)
        self.loss_weight = conf.loss_weight

        camera_model = "simple_radial" if self.conf.estimate_distortions else "pinhole"
        self.camera_model = camera_models[camera_model]

    def _forward(self, data):
        # (B, 3, H, W)
        h, w = data["up_field"].shape[-2:]
        h = h * data["up_field"].new_ones(data["up_field"].shape[0])
        w = w * data["up_field"].new_ones(data["up_field"].shape[0])

        if self.conf.original_weights:
            perspective_image = torch.cat(
                (data["up_field"], torch.sin(data["latitude_field"])), dim=1
            )
        else:
            perspective_image = torch.cat((data["up_field"], data["latitude_field"]), dim=1)

        # (B, #params)
        out = self.backbone({"image": perspective_image})["features"]

        # conversion:
        # roll, pitch: from [-1, 1] to [-pi/2, pi/2]
        # vfov: from vfov / 90 to vfov in radians
        # k1_hat: from [-1, 1] to distortion range
        # conversion = out.new_zeros(out.shape)
        # conversion[:, 2] = torch.pi / 2
        # out *= conversion
        out[:, :2] = out[:, :2] * torch.pi / 2
        out[:, 2] = out[:, 2] * torch.pi / 2

        if self.conf.estimate_distortions:
            low = self.conf.distortion_range[0]
            high = self.conf.distortion_range[1]
            out[:, 3] = (out[:, 3] + 1) / 2 * (high - low) + low

        param_dict = {"vfov": out[:, 2], "height": h, "width": w, "scales": data["scales"]}
        param_dict |= {"k1_hat": out[:, 3]} if self.conf.estimate_distortions else {}
        camera = self.camera_model.from_dict(param_dict)

        gravity = Gravity.from_rp(roll=out[:, 0], pitch=out[:, 1])
        return {"camera": camera, "gravity": gravity}

    def loss(self, pred, data):
        pred_cam, gt_cam = pred["camera"], data["camera"]
        pred_gravity, gt_gravity = pred["gravity"], data["gravity"]

        roll_loss = F.mse_loss(pred_gravity.roll, gt_gravity.roll, reduction="none")
        pitch_loss = F.mse_loss(pred_gravity.pitch, gt_gravity.pitch, reduction="none")
        vfov_loss = F.mse_loss(pred_cam.vfov, gt_cam.vfov, reduction="none")

        roll_loss = roll_loss * self.conf.loss_scales[0]
        pitch_loss = pitch_loss * self.conf.loss_scales[1]
        vfov_loss = vfov_loss * self.conf.loss_scales[2]

        dist_loss = vfov_loss.new_zeros(vfov_loss.shape)
        if self.conf.estimate_distortions:
            dist_loss = F.mse_loss(pred_cam.dist, gt_cam.dist, reduction="none").sum(-1)

        total_loss = self.loss_weight * (roll_loss + pitch_loss + vfov_loss + dist_loss)

        losses = {
            "roll": roll_loss,
            "pitch": pitch_loss,
            "vfov": vfov_loss,
            "distortion": dist_loss,
            "param_total": total_loss,
        }

        return losses, self.metrics(pred, data)

    def metrics(self, pred, data):
        pred_cam, gt_cam = pred["camera"], data["camera"]
        pred_gravity, gt_gravity = pred["gravity"], data["gravity"]

        zeros = pred_cam.new_zeros(pred_cam.f.shape[0])
        return {
            "roll_error": roll_error(pred_gravity, gt_gravity),
            "pitch_error": pitch_error(pred_gravity, gt_gravity),
            "vfov_error": vfov_error(pred_cam, gt_cam),
            "k1_error": dist_error(pred_cam, gt_cam) if self.conf.estimate_distortions else zeros,
        }
