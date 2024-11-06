class ParamNet(torch.nn.Module):
    """Simple interface for ParamNet model."""

    def __init__(self, **conf):
        """Initialize the model with optional config overrides."""
        super().__init__()

        url = "https://polybox.ethz.ch/index.php/s/QDvHFbk9ARiO5hS/download"
        model_dir = f"{torch.hub.get_dir()}/paramnet-cities/"
        state_dict = torch.hub.load_state_dict_from_url(
            url, model_dir, map_location="cpu", file_name="checkpoint.tar"
        )

        state_dict["conf"]["perspective_decoder"]["up_decoder"]["decoder"][
            "name"
        ] = "decoders.persformer_decoder"
        state_dict["conf"]["perspective_decoder"]["latitude_decoder"]["decoder"][
            "name"
        ] = "decoders.persformer_decoder"
        self.model_conf = state_dict["conf"]

        self.conf = OmegaConf.create({**self.model_conf, **conf})

        self.preprocess_conf = {"resize": 320, "edge_divisible_by": 32}
        self.model = get_model(self.conf.name)(self.conf)
        self.model.flexible_load(state_dict["model"])
        self.model.eval()

    @torch.no_grad()
    def calibrate(self, img: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
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

        img_data = ImagePreprocessor({**self.preprocess_conf, **kwargs})(img)

        out = self.model.forward(img_data)

        gravity = out["gravity"]
        camera = out["camera"]

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
